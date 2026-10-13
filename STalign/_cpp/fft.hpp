// Batched multi-dimensional real FFT helpers on top of pocketfft's 1D plans.
//
// Fields live in the STalign velocity layout v[t, i0, i1, (i2), comp]; a
// "field" is one (t, comp) pair.  The half spectrum (r2c along axis 1) is
// stored as spec[t, k0, k1, (k2), comp] with k1 in [0, n1/2].  All line
// transforms are parallelised over lines with the engine's thread pool and
// processed VL lines at a time with pocketfft's SIMD vector type.
//
// pocketfft (header-only, BSD-3-Clause, see third_party/pocketfft/LICENSE.md)
// is vendored from https://github.com/mreineck/pocketfft, branch cpp, c90e55b3.
#pragma once

#define POCKETFFT_NO_MULTITHREADING
#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "pool.hpp"
#include "third_party/pocketfft/pocketfft_hdronly.h"

namespace stalign {

namespace pf = pocketfft::detail;
using cplx = std::complex<double>;
using i64 = int64_t;

#ifndef POCKETFFT_NO_VECTORS
using vdouble = pf::vtype_t<double>;
constexpr int VL = (int)pf::VLEN<double>::val;
#else
constexpr int VL = 1;
#endif

// Generic batched line transforms.  in_off(l)/out_off(l) give the element
// offset of line l; strides are along the transformed axis (in elements of
// double for real arrays, of complex for complex arrays).

template <class InOff, class OutOff>
void r2c_lines(ThreadPool& pool, const pf::pocketfft_r<double>& plan, i64 n, const double* in,
               i64 istr, InOff in_off, cplx* out, i64 ostr, OutOff out_off, i64 nlines,
               i64 grain) {
  const i64 ngroups = (nlines + VL - 1) / VL;
  const i64 ntask = (ngroups + grain - 1) / grain;
  pool.parallel_for(ntask, [&](i64 task) {
#ifndef POCKETFFT_NO_VECTORS
    std::vector<vdouble> vbuf(n);
#endif
    std::vector<double> sbuf(n);
    const i64 g0 = task * grain, g1 = std::min(ngroups, g0 + grain);
    for (i64 g = g0; g < g1; ++g) {
      const i64 l0 = g * VL;
#ifndef POCKETFFT_NO_VECTORS
      if (l0 + VL <= nlines) {
        i64 io[VL], oo[VL];
        bool contig = true;
        for (int j = 0; j < VL; ++j) {
          io[j] = in_off(l0 + j);
          oo[j] = out_off(l0 + j);
          contig = contig && io[j] == io[0] + j && oo[j] == oo[0] + j;
        }
        if (contig) {  // lanes are adjacent in memory: whole-vector loads / stores
          for (i64 k = 0; k < n; ++k) std::memcpy(&vbuf[k], in + io[0] + k * istr, sizeof(vdouble));
        } else {
          for (i64 k = 0; k < n; ++k)
            for (int j = 0; j < VL; ++j) vbuf[k][j] = in[io[j] + k * istr];
        }
        plan.exec(vbuf.data(), 1.0, true);
        if (contig) {
          double tmp[2 * VL];
          cplx* o = out + oo[0];
          for (int j = 0; j < VL; ++j) { tmp[2 * j] = vbuf[0][j]; tmp[2 * j + 1] = 0.0; }
          std::memcpy(o, tmp, sizeof(tmp));
          i64 i = 1, ii = 1;
          for (; i < n - 1; i += 2, ++ii) {
            for (int j = 0; j < VL; ++j) { tmp[2 * j] = vbuf[i][j]; tmp[2 * j + 1] = vbuf[i + 1][j]; }
            std::memcpy(o + ii * ostr, tmp, sizeof(tmp));
          }
          if (i < n) {
            for (int j = 0; j < VL; ++j) { tmp[2 * j] = vbuf[i][j]; tmp[2 * j + 1] = 0.0; }
            std::memcpy(o + ii * ostr, tmp, sizeof(tmp));
          }
          continue;
        }
        for (int j = 0; j < VL; ++j) {
          cplx* o = out + oo[j];
          o[0] = cplx(vbuf[0][j], 0.0);
          i64 i = 1, ii = 1;
          for (; i < n - 1; i += 2, ++ii) o[ii * ostr] = cplx(vbuf[i][j], vbuf[i + 1][j]);
          if (i < n) o[ii * ostr] = cplx(vbuf[i][j], 0.0);
        }
        continue;
      }
#endif
      for (i64 l = l0; l < std::min(nlines, l0 + VL); ++l) {
        const double* src = in + in_off(l);
        for (i64 k = 0; k < n; ++k) sbuf[k] = src[k * istr];
        plan.exec(sbuf.data(), 1.0, true);
        cplx* o = out + out_off(l);
        o[0] = cplx(sbuf[0], 0.0);
        i64 i = 1, ii = 1;
        for (; i < n - 1; i += 2, ++ii) o[ii * ostr] = cplx(sbuf[i], sbuf[i + 1]);
        if (i < n) o[ii * ostr] = cplx(sbuf[i], 0.0);
      }
    }
  });
}

struct StoreOp {
  inline void operator()(double& dst, double val) const { dst = val; }
};

// out[...] is updated with op(out_elem, value) (plain store by default)
template <class InOff, class OutOff, class Op = StoreOp>
void c2r_lines(ThreadPool& pool, const pf::pocketfft_r<double>& plan, i64 n, const cplx* in,
               i64 istr, InOff in_off, double* out, i64 ostr, OutOff out_off, i64 nlines,
               double fct, i64 grain, Op op = Op()) {
  const i64 ngroups = (nlines + VL - 1) / VL;
  const i64 ntask = (ngroups + grain - 1) / grain;
  pool.parallel_for(ntask, [&](i64 task) {
#ifndef POCKETFFT_NO_VECTORS
    std::vector<vdouble> vbuf(n);
#endif
    std::vector<double> sbuf(n);
    const i64 g0 = task * grain, g1 = std::min(ngroups, g0 + grain);
    for (i64 g = g0; g < g1; ++g) {
      const i64 l0 = g * VL;
#ifndef POCKETFFT_NO_VECTORS
      if (l0 + VL <= nlines) {
        i64 io[VL], oo[VL];
        bool contig = true;
        for (int j = 0; j < VL; ++j) {
          io[j] = in_off(l0 + j);
          oo[j] = out_off(l0 + j);
          contig = contig && io[j] == io[0] + j && oo[j] == oo[0] + j;
        }
        if (contig) {
          double tmp[2 * VL];
          const cplx* s = in + io[0];
          std::memcpy(tmp, s, sizeof(tmp));
          for (int j = 0; j < VL; ++j) vbuf[0][j] = tmp[2 * j];
          i64 i = 1, ii = 1;
          for (; i < n - 1; i += 2, ++ii) {
            std::memcpy(tmp, s + ii * istr, sizeof(tmp));
            for (int j = 0; j < VL; ++j) { vbuf[i][j] = tmp[2 * j]; vbuf[i + 1][j] = tmp[2 * j + 1]; }
          }
          if (i < n) {
            std::memcpy(tmp, s + ii * istr, sizeof(tmp));
            for (int j = 0; j < VL; ++j) vbuf[i][j] = tmp[2 * j];
          }
          plan.exec(vbuf.data(), fct, false);
          double* o = out + oo[0];
          for (i64 k = 0; k < n; ++k) {
            double* dst = o + k * ostr;
            for (int j = 0; j < VL; ++j) op(dst[j], vbuf[k][j]);
          }
          continue;
        }
        for (int j = 0; j < VL; ++j) {
          const cplx* s = in + io[j];
          vbuf[0][j] = s[0].real();
          i64 i = 1, ii = 1;
          for (; i < n - 1; i += 2, ++ii) {
            vbuf[i][j] = s[ii * istr].real();
            vbuf[i + 1][j] = s[ii * istr].imag();
          }
          if (i < n) vbuf[i][j] = s[ii * istr].real();
        }
        plan.exec(vbuf.data(), fct, false);
        for (i64 k = 0; k < n; ++k)
          for (int j = 0; j < VL; ++j) op(out[oo[j] + k * ostr], vbuf[k][j]);
        continue;
      }
#endif
      for (i64 l = l0; l < std::min(nlines, l0 + VL); ++l) {
        const cplx* s = in + in_off(l);
        sbuf[0] = s[0].real();
        i64 i = 1, ii = 1;
        for (; i < n - 1; i += 2, ++ii) {
          sbuf[i] = s[ii * istr].real();
          sbuf[i + 1] = s[ii * istr].imag();
        }
        if (i < n) sbuf[i] = s[ii * istr].real();
        plan.exec(sbuf.data(), fct, false);
        double* o = out + out_off(l);
        for (i64 k = 0; k < n; ++k) op(o[k * ostr], sbuf[k]);
      }
    }
  });
}

template <class Off>
void c2c_lines(ThreadPool& pool, const pf::pocketfft_c<double>& plan, i64 n, cplx* data,
               i64 str, Off off, i64 nlines, bool fwd, double fct, i64 grain) {
  const i64 ngroups = (nlines + VL - 1) / VL;
  const i64 ntask = (ngroups + grain - 1) / grain;
  pool.parallel_for(ntask, [&](i64 task) {
#ifndef POCKETFFT_NO_VECTORS
    std::vector<pf::cmplx<vdouble>> vbuf(n);
#endif
    std::vector<pf::cmplx<double>> sbuf(n);
    const i64 g0 = task * grain, g1 = std::min(ngroups, g0 + grain);
    for (i64 g = g0; g < g1; ++g) {
      const i64 l0 = g * VL;
#ifndef POCKETFFT_NO_VECTORS
      if (l0 + VL <= nlines) {
        i64 oo[VL];
        for (int j = 0; j < VL; ++j) oo[j] = off(l0 + j);
        for (i64 k = 0; k < n; ++k)
          for (int j = 0; j < VL; ++j) {
            const cplx& z = data[oo[j] + k * str];
            vbuf[k].r[j] = z.real();
            vbuf[k].i[j] = z.imag();
          }
        plan.exec(vbuf.data(), fct, fwd);
        for (i64 k = 0; k < n; ++k)
          for (int j = 0; j < VL; ++j) data[oo[j] + k * str] = cplx(vbuf[k].r[j], vbuf[k].i[j]);
        continue;
      }
#endif
      for (i64 l = l0; l < std::min(nlines, l0 + VL); ++l) {
        cplx* d = data + off(l);
        for (i64 k = 0; k < n; ++k) sbuf[k] = pf::cmplx<double>(d[k * str].real(), d[k * str].imag());
        plan.exec(sbuf.data(), fct, fwd);
        for (i64 k = 0; k < n; ++k) d[k * str] = cplx(sbuf[k].r, sbuf[k].i);
      }
    }
  });
}

inline i64 ntasks_for(i64 nlines, i64 grain) {
  const i64 ngroups = (nlines + VL - 1) / VL;
  return (ngroups + grain - 1) / grain;
}

// Fused per-line filter along one c2c axis (one task owns whole lines):
//   z_k = pre(task, l, k); Z = FFT(z); Z_k = mid(task, l, k, Z_k); z = IFFT(Z) (unnormalised);
//   post(task, l, k, z_k).
// Lines are visited in a fixed order per task, so per-task reductions done in the
// hooks are deterministic (independent of the number of threads).
template <class Pre, class Mid, class Post>
void c2c_lines_filter(ThreadPool& pool, const pf::pocketfft_c<double>& plan, i64 n, i64 nlines,
                      i64 grain, Pre pre, Mid mid, Post post) {
  const i64 ngroups = (nlines + VL - 1) / VL;
  const i64 ntask = (ngroups + grain - 1) / grain;
  pool.parallel_for(ntask, [&](i64 task) {
#ifndef POCKETFFT_NO_VECTORS
    std::vector<pf::cmplx<vdouble>> vbuf(n);
#endif
    std::vector<pf::cmplx<double>> sbuf(n);
    const i64 g0 = task * grain, g1 = std::min(ngroups, g0 + grain);
    for (i64 g = g0; g < g1; ++g) {
      const i64 l0 = g * VL;
#ifndef POCKETFFT_NO_VECTORS
      if (l0 + VL <= nlines) {
        for (i64 k = 0; k < n; ++k)
          for (int j = 0; j < VL; ++j) {
            const cplx z = pre(task, l0 + j, k);
            vbuf[k].r[j] = z.real();
            vbuf[k].i[j] = z.imag();
          }
        plan.exec(vbuf.data(), 1.0, true);
        for (i64 k = 0; k < n; ++k)
          for (int j = 0; j < VL; ++j) {
            const cplx z = mid(task, l0 + j, k, cplx(vbuf[k].r[j], vbuf[k].i[j]));
            vbuf[k].r[j] = z.real();
            vbuf[k].i[j] = z.imag();
          }
        plan.exec(vbuf.data(), 1.0, false);
        for (i64 k = 0; k < n; ++k)
          for (int j = 0; j < VL; ++j) post(task, l0 + j, k, cplx(vbuf[k].r[j], vbuf[k].i[j]));
        continue;
      }
#endif
      for (i64 l = l0; l < std::min(nlines, l0 + VL); ++l) {
        for (i64 k = 0; k < n; ++k) {
          const cplx z = pre(task, l, k);
          sbuf[k] = pf::cmplx<double>(z.real(), z.imag());
        }
        plan.exec(sbuf.data(), 1.0, true);
        for (i64 k = 0; k < n; ++k) {
          const cplx z = mid(task, l, k, cplx(sbuf[k].r, sbuf[k].i));
          sbuf[k] = pf::cmplx<double>(z.real(), z.imag());
        }
        plan.exec(sbuf.data(), 1.0, false);
        for (i64 k = 0; k < n; ++k) post(task, l, k, cplx(sbuf[k].r, sbuf[k].i));
      }
    }
  });
}

// Spectral helper for velocity-like fields v[t, i0, i1, (i2), c] (c = 0..D-1).
// The half spectrum (r2c along axis 1) mirrors that layout with the component fastest:
//     spec[t, k0, k1, (k2), c],  k1 in [0, n1/2]
// so the VL lanes of every line group are adjacent in memory in every pass
// (input and output), and one task streams through contiguous slabs.
// Elementwise kernels (K, LL) are indexed by e = (k0*h1 + k1)*n2 + k2.
// All transforms are unnormalised (like pocketfft's, fct = 1).
struct FieldFFT {
  i64 n0 = 0, n1 = 0, n2 = 1, h1 = 0, D = 0, nt = 0, F = 0;
  std::unique_ptr<pf::pocketfft_r<double>> r1;
  std::unique_ptr<pf::pocketfft_c<double>> c0, c2;

  void init(i64 n0_, i64 n1_, i64 n2_, i64 D_, i64 nt_) {
    n0 = n0_; n1 = n1_; n2 = n2_; D = D_; nt = nt_;
    h1 = n1 / 2 + 1;
    F = nt * D;
    r1.reset(new pf::pocketfft_r<double>((size_t)n1));
    c0.reset(new pf::pocketfft_c<double>((size_t)n0));
    if (n2 > 1) c2.reset(new pf::pocketfft_c<double>((size_t)n2));
  }
  i64 spec_size() const { return F * n0 * h1 * n2; }
  i64 spec_field() const { return n0 * h1 * n2; }  // number of distinct kernel entries

  // forward FFT over all spatial axes of every field: spec = fftn(field)
  void forward(ThreadPool& pool, const double* v, cplx* spec) const {
    r2c1(pool, v, spec);
    c2c0(pool, spec, true);
    axis2(pool, spec, true);
  }

  // inverse over all spatial axes: out = fct * ifftn_unnormalised(spec); spec is destroyed
  void inverse(ThreadPool& pool, cplx* spec, double* out, double fct) const {
    axis2(pool, spec, false);
    c2c0(pool, spec, false);
    c2r1(pool, spec, out, fct, StoreOp());
  }

  // r2c along axis 1.  Lines l = (r, m) with r = t*n0 + i0 and m = i2*D + c.
  void r2c1(ThreadPool& pool, const double* v, cplx* spec) const {
    const i64 nl = F * n0 * n2;
    const i64 SL = n2 * D;  // lines per (t, i0) slab
    auto in_off = [=](i64 l) { return (l / SL) * n1 * SL + l % SL; };
    auto out_off = [=](i64 l) { return (l / SL) * h1 * SL + l % SL; };
    r2c_lines(pool, *r1, n1, v, SL, in_off, spec, SL, out_off, nl, grain(n1, nl));
  }

  // c2c along axis 0, in place.  Lines (t, k1, (i2 or k2), c).
  void c2c0(ThreadPool& pool, cplx* spec, bool fwd) const {
    const i64 PL = h1 * n2 * D;  // lines per t
    const i64 nl0 = nt * PL;
    auto off0 = [=](i64 l) { return (l / PL) * n0 * PL + l % PL; };
    c2c_lines(pool, *c0, n0, spec, PL, off0, nl0, fwd, 1.0, grain(n0, nl0));
  }

  // c2c along axis 2 (3D only), in place.  Lines (t, k0, k1, c), stride D.
  void axis2(ThreadPool& pool, cplx* spec, bool fwd) const {
    if (n2 <= 1) return;
    const i64 nl = F * n0 * h1;
    auto off = [=](i64 l) { return (l / D) * n2 * D + l % D; };
    c2c_lines(pool, *c2, n2, spec, D, off, nl, fwd, 1.0, grain(n2, nl));
  }

  // c2r along axis 1 into the velocity layout: op(out_elem, fct * value)
  template <class Op>
  void c2r1(ThreadPool& pool, const cplx* spec, double* out, double fct, Op op) const {
    const i64 nl = F * n0 * n2;
    const i64 SL = n2 * D;
    auto in_off = [=](i64 l) { return (l / SL) * h1 * SL + l % SL; };
    auto out_off = [=](i64 l) { return (l / SL) * n1 * SL + l % SL; };
    c2r_lines(pool, *r1, n1, spec, SL, in_off, out, SL, out_off, nl, fct, grain(n1, nl), op);
  }

  static i64 grain(i64 n, i64 nlines) {
    // aim for ~ 4-8k complex points per task, but at least one vector group
    i64 g = std::max<i64>(1, 4096 / std::max<i64>(1, n * VL));
    const i64 ngroups = (nlines + VL - 1) / VL;
    // keep at least ~64 tasks when possible
    g = std::min<i64>(g, std::max<i64>(1, ngroups / 64));
    return g;
  }
};

}  // namespace stalign
