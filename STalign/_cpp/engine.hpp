// Fused CPU engine for STalign's LDDMM (D = 2) and LDDMM_3D_to_slice (D = 3).
//
// One iteration of the torch implementation is reproduced in float64:
//   pass 1 (parallel over chunks of target rows): affine sample points,
//          backward integration of the velocity field (storing X^(t) for all t),
//          image interpolation (border padding) + spatial gradient, and the
//          weighted reductions for the contrast normal equations (BB, BJ).
//   solve the (1+C_I)x(1+C_I) system for the contrast coefficients.
//   pass 2 (parallel over the same chunks): fAI, EM, dE/dAI, backprop through
//          the nt velocity steps with splats into chunk-local bounding-box
//          buffers, accumulation of dE/dAi.
//   deterministic reduction of the chunk buffers into dE/dv, points term,
//   affine gradient, FFT smoothing (pocketfft), parameter update, EM weights.
//
// The landmark term (2D) maps the landmarks by forward Euler steps or, by default,
// exactly by the inverse of the backward map (fixed-point solve of each implicit
// step, implicit-function gradient).  The weights come from the original's E-step
// or from the mixture EM (Params::mixture).  Both are described in
// STalign/_lddmm_common.py (class Problem).
//
// The regulariser is ER = DV / (2 sigmaR^2 N) sum_k LL_k |fftn(v)_k|^2 with the
// FFT over all spatial axes (N = number of velocity grid points).  Since the
// smoothing kernel is K = 1/LL, the smoothed ER gradient is exactly
// DV/sigmaR^2 * v, so the velocity update needs one FFT round trip (of dE/dv
// without ER) per iteration.
//
// Every reduction is done per chunk and combined in chunk order, so results do
// not depend on the number of threads.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "fft.hpp"
#include "pool.hpp"

namespace stalign {

// Limits of the engine (sizes of per-chunk stack buffers).  The Python adapter
// checks them first and uses the torch backend for larger problems.
constexpr int MAX_CI = 31;  // image channels
constexpr int MAX_CJ = 64;  // target channels
constexpr int MAX_NT = 64;  // velocity time steps

// Exact landmark map: a fixed-point solve stops when a sweep moves no coordinate of the
// landmark by more than LANDMARK_TOL velocity grid cells (or a few ulps of the grid's
// coordinates), after at most LANDMARK_MAX_ITER sweeps.  (Same values in the torch engine.)
constexpr double LANDMARK_TOL = 1e-10;
constexpr int LANDMARK_MAX_ITER = 100;
// Mixture EM: floor of the estimated matching variance.
constexpr double VARIANCE_FLOOR = 1e-12;

#if defined(__GNUC__) || defined(__clang__)
#define STALIGN_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define STALIGN_NOINLINE __attribute__((noinline))
#else
#define STALIGN_UNLIKELY(x) (x)
#define STALIGN_NOINLINE __declspec(noinline)
#endif

// A regular grid with D axes, as used by STalign.interp (torch grid_sample with
// align_corners=True): only the first and last coordinate of each axis matter.
template <int D>
struct Grid {
  i64 n[D];
  double x0[D];
  double ext[D];    // x_last - x_first
  double half[D];   // (n-1)/2
  double scale[D];  // (n-1) / (x_last - x_first): d u / d X
  i64 str[D];       // row-major point strides
  i64 npts = 0;

  // Continuous index exactly as STalign.interp + grid_sample compute it:
  // coord = ((X - x0) / ext) * 2 - 1 ; u = (coord + 1) * ((n-1)/2).
  // (The 3D variant's ((coord+1)/2)*(n-1) is bitwise identical.)  Reproducing the
  // rounding matters only when X sits exactly on a grid line (e.g. identity start),
  // where the one-sided image gradient depends on which cell floor(u) selects.
  inline double uindex_exact(const double* X, int d) const {
    const double c = ((X[d] - x0[d]) / ext[d]) * 2.0 - 1.0;
    return (c + 1.0) * half[d];
  }
  // Fast version (no division); differs from uindex_exact by a few ulps.  Callers
  // switch to uindex_exact when the result is within NEAR of a grid line, so the
  // selected cell always matches torch.
  inline double uindex(const double* X, int d) const { return (X[d] - x0[d]) * scale[d]; }
  static constexpr double NEAR = 1e-9;

  void init(const i64* n_, const double* x0_, const double* xl_, const char* name) {
    npts = 1;
    for (int d = D - 1; d >= 0; --d) {
      n[d] = n_[d];
      str[d] = npts;
      npts *= n[d];
    }
    for (int d = 0; d < D; ++d) {
      if (n[d] < 2)
        throw std::invalid_argument(std::string(name) + " needs at least 2 samples per axis");
      x0[d] = x0_[d];
      ext[d] = xl_[d] - x0_[d];
      half[d] = double(n[d] - 1) / 2.0;
      scale[d] = double(n[d] - 1) / ext[d];
    }
  }
};

// cold path of vsetup: index with torch's exact rounding
template <int D>
STALIGN_NOINLINE static void vexact(const Grid<D>& g, const double* X, int d, i64& i, double& f) {
  const double hi = double(g.n[d]) + 1.0;
  double u = g.uindex_exact(X, d);
  if (!(u > -2.0)) u = -2.0;
  if (u > hi) u = hi;
  const double fl = std::floor(u);
  i = (i64)fl;
  f = u - fl;
}

// A sample point of a velocity field (zeros padding): cell, fractions and which
// corners lie inside the grid.
template <int D>
struct VSample {
  i64 i0[D];
  double f[D];
  bool ok0[D], ok1[D];
  bool interior;
};

// bit d of corner c (corner 0 = lower corner along every axis, axis 0 most significant)
template <int D>
static inline int cbit(int c, int d) {
  return (c >> (D - 1 - d)) & 1;
}

// zeros padding: returns false if no corner is inside the grid
template <int D>
static inline bool vsetup(const Grid<D>& g, const double* X, VSample<D>& s) {
  bool any = true, interior = true;
  for (int d = 0; d < D; ++d) {
    const double hi = double(g.n[d]) + 1.0;
    double u = g.uindex(X, d);
    if (!(u > -2.0)) u = -2.0;  // also maps NaN out of the grid
    if (u > hi) u = hi;
    const double fl = std::floor(u);
    i64 i = (i64)fl;
    double f = u - fl;
    if (STALIGN_UNLIKELY(f < Grid<D>::NEAR || f > 1.0 - Grid<D>::NEAR))
      vexact(g, X, d, i, f);  // near a grid line: torch rounding
    s.i0[d] = i;
    s.f[d] = f;
    s.ok0[d] = (i >= 0) && (i < g.n[d]);
    s.ok1[d] = (i + 1 >= 0) && (i + 1 < g.n[d]);
    any = any && (s.ok0[d] || s.ok1[d]);
    interior = interior && s.ok0[d] && s.ok1[d];
  }
  s.interior = interior;
  return any;
}

// value of the velocity field vt (layout (n..., D)) at a sample
template <int D>
static inline void vvalue(const Grid<D>& g, const double* __restrict vt, const VSample<D>& s,
                          double* out) {
  constexpr int NC = 1 << D;
  double W[D][2];
  i64 base = 0;
  for (int d = 0; d < D; ++d) {
    W[d][1] = s.f[d];
    W[d][0] = 1.0 - s.f[d];
    base += s.i0[d] * g.str[d];
  }
  for (int k = 0; k < D; ++k) out[k] = 0.0;
  if (s.interior) {
    for (int c = 0; c < NC; ++c) {
      i64 off = base;
      double w = 1.0;
      for (int d = 0; d < D; ++d) {
        const int b = cbit<D>(c, d);
        off += b * g.str[d];
        w *= W[d][b];
      }
      const double* p = vt + off * D;
      for (int k = 0; k < D; ++k) out[k] += w * p[k];
    }
  } else {
    for (int c = 0; c < NC; ++c) {
      i64 off = base;
      double w = 1.0;
      bool inb = true;
      for (int d = 0; d < D; ++d) {
        const int b = cbit<D>(c, d);
        inb = inb && (b ? s.ok1[d] : s.ok0[d]);
        off += b * g.str[d];
        w *= W[d][b];
      }
      if (!inb) continue;
      const double* p = vt + off * D;
      for (int k = 0; k < D; ++k) out[k] += w * p[k];
    }
  }
}

// Backprop through one velocity sample.  q = upstream gradient / nt.
// Adds sgn * w_c * q to the splat buffer (if SPLAT) and returns
// JTq[d] = sum_k q_k d v_k / d X_d.
template <int D, bool SPLAT>
static inline void vback(const Grid<D>& g, const double* __restrict vt, const VSample<D>& s,
                         const double* q, double sgn, double* __restrict lbuf, const i64* lo,
                         const i64* lstr, double* JTq) {
  constexpr int NC = 1 << D;
  double W[D][2];
  i64 base = 0, lbase = 0;
  for (int d = 0; d < D; ++d) {
    W[d][1] = s.f[d];
    W[d][0] = 1.0 - s.f[d];
    base += s.i0[d] * g.str[d];
    if (SPLAT) lbase += (s.i0[d] - lo[d]) * lstr[d];
  }
  double acc[D];
  for (int d = 0; d < D; ++d) acc[d] = 0.0;
  double sq[D];
  for (int k = 0; k < D; ++k) sq[k] = sgn * q[k];
  for (int c = 0; c < NC; ++c) {
    i64 off = base, loff = lbase;
    bool inb = true;
    for (int d = 0; d < D; ++d) {
      const int b = cbit<D>(c, d);
      if (!s.interior) inb = inb && (b ? s.ok1[d] : s.ok0[d]);
      off += b * g.str[d];
      if (SPLAT) loff += b * lstr[d];
    }
    if (!inb) continue;
    // weights and d weight / d u_d
    double w = 1.0;
    double dw[D];
    for (int d = 0; d < D; ++d) dw[d] = cbit<D>(c, d) ? 1.0 : -1.0;
    for (int e = 0; e < D; ++e) {
      const double We = W[e][cbit<D>(c, e)];
      w *= We;
      for (int d = 0; d < D; ++d)
        if (d != e) dw[d] *= We;
    }
    const double* p = vt + off * D;
    double sc = 0.0;
    for (int k = 0; k < D; ++k) sc += q[k] * p[k];
    for (int d = 0; d < D; ++d) acc[d] += dw[d] * sc;
    if (SPLAT) {
      double* lb = lbuf + loff * D;
      for (int k = 0; k < D; ++k) lb[k] += w * sq[k];
    }
  }
  for (int d = 0; d < D; ++d) JTq[d] = acc[d] * g.scale[d];
}

// Spatial Jacobian of a velocity sample, Jac[k][d] = d v_k / d X_d (only corners inside
// the grid contribute, as in vback).
template <int D>
static inline void vjacobian(const Grid<D>& g, const double* __restrict vt, const VSample<D>& s,
                             double (&Jac)[D][D]) {
  constexpr int NC = 1 << D;
  double W[D][2];
  i64 base = 0;
  for (int d = 0; d < D; ++d) {
    W[d][1] = s.f[d];
    W[d][0] = 1.0 - s.f[d];
    base += s.i0[d] * g.str[d];
  }
  for (int k = 0; k < D; ++k)
    for (int d = 0; d < D; ++d) Jac[k][d] = 0.0;
  for (int c = 0; c < NC; ++c) {
    i64 off = base;
    bool inb = true;
    for (int d = 0; d < D; ++d) {
      const int b = cbit<D>(c, d);
      if (!s.interior) inb = inb && (b ? s.ok1[d] : s.ok0[d]);
      off += b * g.str[d];
    }
    if (!inb) continue;
    double dw[D];
    for (int d = 0; d < D; ++d) dw[d] = cbit<D>(c, d) ? 1.0 : -1.0;
    for (int e = 0; e < D; ++e) {
      const double We = W[e][cbit<D>(c, e)];
      for (int d = 0; d < D; ++d)
        if (d != e) dw[d] *= We;
    }
    const double* p = vt + off * D;
    for (int k = 0; k < D; ++k)
      for (int d = 0; d < D; ++d) Jac[k][d] += dw[d] * p[k];
  }
  for (int k = 0; k < D; ++k)
    for (int d = 0; d < D; ++d) Jac[k][d] *= g.scale[d];
}

// Splat accumulator: consecutive target pixels usually fall into the same
// velocity cell (J is much finer than v), so interior splats are summed in a
// small per-cell accumulator and written to the chunk buffer when the cell changes.
template <int D>
struct SplatAcc {
  static constexpr int NC = 1 << D;
  i64 lbase = -1;  // local (box) point offset of corner 0 of the cached cell
  // zeroed before use whenever lbase is set (vback_acc), and read only when lbase >= 0.  GCC
  // reports a false-positive -Wmaybe-uninitialized here, so it gets an initializer; clang does
  // not, and there the initializer makes pass 2 about 4% slower (2D, Apple M1).
#if defined(__GNUC__) && !defined(__clang__)
  double acc[NC][D] = {};
#else
  double acc[NC][D];
#endif
  void flush(double* __restrict lbuf, const i64* lstr) {
    if (lbase < 0) return;
    for (int c = 0; c < NC; ++c) {
      i64 loff = lbase;
      for (int d = 0; d < D; ++d) loff += cbit<D>(c, d) * lstr[d];
      double* lb = lbuf + loff * D;
      for (int k = 0; k < D; ++k) lb[k] += acc[c][k];
    }
    lbase = -1;
  }
};

// vback<D, true> with sgn = -1 and cell-cached splats
template <int D>
static inline void vback_acc(const Grid<D>& g, const double* __restrict vt, const VSample<D>& s,
                             const double* q, double* __restrict lbuf, const i64* lo,
                             const i64* lstr, double* JTq, SplatAcc<D>& sa) {
  if (!s.interior) {
    vback<D, true>(g, vt, s, q, -1.0, lbuf, lo, lstr, JTq);
    return;
  }
  constexpr int NC = 1 << D;
  double W[D][2];
  i64 base = 0, lbase = 0;
  for (int d = 0; d < D; ++d) {
    W[d][1] = s.f[d];
    W[d][0] = 1.0 - s.f[d];
    base += s.i0[d] * g.str[d];
    lbase += (s.i0[d] - lo[d]) * lstr[d];
  }
  if (lbase != sa.lbase) {
    sa.flush(lbuf, lstr);
    sa.lbase = lbase;
    for (int c = 0; c < NC; ++c)
      for (int k = 0; k < D; ++k) sa.acc[c][k] = 0.0;
  }
  double acc[D];
  for (int d = 0; d < D; ++d) acc[d] = 0.0;
  double sq[D];
  for (int k = 0; k < D; ++k) sq[k] = -q[k];
  for (int c = 0; c < NC; ++c) {
    i64 off = base;
    for (int d = 0; d < D; ++d) off += cbit<D>(c, d) * g.str[d];
    double w = 1.0;
    double dw[D];
    for (int d = 0; d < D; ++d) dw[d] = cbit<D>(c, d) ? 1.0 : -1.0;
    for (int e = 0; e < D; ++e) {
      const double We = W[e][cbit<D>(c, e)];
      w *= We;
      for (int d = 0; d < D; ++d)
        if (d != e) dw[d] *= We;
    }
    const double* p = vt + off * D;
    double sc = 0.0;
    for (int k = 0; k < D; ++k) sc += q[k] * p[k];
    for (int d = 0; d < D; ++d) acc[d] += dw[d] * sc;
    for (int k = 0; k < D; ++k) sa.acc[c][k] += w * sq[k];
  }
  for (int d = 0; d < D; ++d) JTq[d] = acc[d] * g.scale[d];
}

// Image sample with border padding (all CI channels) and its spatial gradient.
// Along an axis where the point is clamped the gradient is zero, as in torch.
template <int D>
static inline void isample(const Grid<D>& g, const double* __restrict I, i64 NI, int CI,
                           const double* X, double* AI, double* dAI) {
  constexpr int NC = 1 << D;
  double W[D][2], gm[D];
  i64 base = 0;
  for (int d = 0; d < D; ++d) {
    const double mx = double(g.n[d] - 1);
    double u = g.uindex(X, d);
    // near a grid line or a border: use torch's rounding (decides cell / clamping)
    if (!(u > Grid<D>::NEAR) || !(u < mx - Grid<D>::NEAR)) {
      u = g.uindex_exact(X, d);
    } else {
      const double fr = u - (double)(i64)u;
      if (fr < Grid<D>::NEAR || fr > 1.0 - Grid<D>::NEAR) u = g.uindex_exact(X, d);
    }
    double m = g.scale[d];
    if (!(u > 0.0)) {
      u = 0.0;
      m = 0.0;
    } else if (u >= mx) {
      u = mx;
      m = 0.0;
    }
    i64 i = (i64)u;  // u >= 0: truncation == floor
    double f = u - (double)i;
    if (i >= g.n[d] - 1) {  // u == n-1: identical value from the (n-2, n-1) cell with f = 1
      i = g.n[d] - 2;
      f = 1.0;
    }
    W[d][0] = 1.0 - f;
    W[d][1] = f;
    gm[d] = m;
    base += i * g.str[d];
  }
  i64 off[NC];
  double w[NC], dw[NC][D];
  for (int c = 0; c < NC; ++c) {
    off[c] = base;
    w[c] = 1.0;
    for (int d = 0; d < D; ++d) {
      off[c] += cbit<D>(c, d) * g.str[d];
      dw[c][d] = cbit<D>(c, d) ? 1.0 : -1.0;
    }
    for (int e = 0; e < D; ++e) {
      const double We = W[e][cbit<D>(c, e)];
      w[c] *= We;
      for (int d = 0; d < D; ++d)
        if (d != e) dw[c][d] *= We;
    }
  }
  for (int ch = 0; ch < CI; ++ch) {
    const double* Ic = I + ch * NI;
    double val = 0.0, dv[D];
    for (int d = 0; d < D; ++d) dv[d] = 0.0;
    for (int c = 0; c < NC; ++c) {
      const double x = Ic[off[c]];
      val += w[c] * x;
      for (int d = 0; d < D; ++d) dv[d] += dw[c][d] * x;
    }
    AI[ch] = val;
    for (int d = 0; d < D; ++d) dAI[ch * D + d] = dv[d] * gm[d];
  }
}

// small dense solve with partial pivoting: A (n x n) X = B (n x m), in place
static inline void small_solve(int n, int m, double* A, double* B) {
  for (int k = 0; k < n; ++k) {
    int p = k;
    double best = std::fabs(A[k * n + k]);
    for (int i = k + 1; i < n; ++i)
      if (std::fabs(A[i * n + k]) > best) {
        best = std::fabs(A[i * n + k]);
        p = i;
      }
    if (p != k) {
      for (int j = 0; j < n; ++j) std::swap(A[k * n + j], A[p * n + j]);
      for (int j = 0; j < m; ++j) std::swap(B[k * m + j], B[p * m + j]);
    }
    const double piv = A[k * n + k];
    for (int i = k + 1; i < n; ++i) {
      const double l = A[i * n + k] / piv;
      if (l == 0.0) continue;
      for (int j = k; j < n; ++j) A[i * n + j] -= l * A[k * n + j];
      for (int j = 0; j < m; ++j) B[i * m + j] -= l * B[k * m + j];
    }
  }
  for (int k = n - 1; k >= 0; --k) {
    for (int j = 0; j < m; ++j) {
      double s = B[k * m + j];
      for (int i = k + 1; i < n; ++i) s -= A[k * n + i] * B[i * m + j];
      B[k * m + j] = s / A[k * n + k];
    }
  }
}

// determinant of a 2x2 or 3x3 matrix (row-major), by the cofactor formula
static inline double small_det(int n, const double* A) {
  if (n == 2) return A[0] * A[3] - A[1] * A[2];
  return A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) +
         A[2] * (A[3] * A[7] - A[4] * A[6]);
}

static inline void small_inverse(int n, const double* A, double* Ai) {
  std::vector<double> a(A, A + n * n), b(n * n, 0.0);
  for (int i = 0; i < n; ++i) b[i * n + i] = 1.0;
  small_solve(n, n, a.data(), b.data());
  std::copy(b.begin(), b.end(), Ai);
}

struct Params {
  double sigmaM = 1, sigmaA = 5, sigmaB = 2, sigmaR = 5e5, sigmaP = 20;
  double epL = 2e-8, epT = 0.2, epV = 2e3;
  int diffeo_start = 0;
  double DV = 1;  // volume of one velocity grid cell, prod(dv)
  // affine step control: per-parameter step multipliers m in [step_floor, step_cap],
  // times step_shrink when the parameter's gradient changes sign, else times step_grow
  bool affine_step_control = false;
  double step_shrink = 0.5, step_grow = 1.05, step_floor = 1e-3, step_cap = 1.0;
  // landmark term: the exact inverse of the backward map (else forward Euler steps)
  bool landmark_exact = true;
  // mixture EM instead of the original weight updates: E-steps after iterations
  // mix_start, mix_start + mix_every, ...; the classes present besides matching; which class
  // means are estimated (the initial means are Inputs::muA / muB); the lower bound of the
  // matching prior and the initial priors (matching, artifact, background)
  bool mixture = false;
  int mix_start = 50, mix_every = 5;
  bool mix_classA = false, mix_classB = true;
  bool mix_est_muA = true, mix_est_muB = false;
  double mix_pi_min = 0.5;
  double mix_pi_init[3] = {0.9, 0.05, 0.05};
};

// Problem description for Engine<D>.  Arrays are float64 in C order.  I and J are
// referenced, not copied, and must outlive the engine; everything else is copied.
template <int D>
struct Inputs {
  int nt = 0, CI = 0, CJ = 0;
  i64 nI[D] = {}, nJ[D] = {}, nV[D] = {};  // spatial shapes of I, J and the velocity grid
  double xI0[D] = {}, xIl[D] = {};          // first / last coordinate of each axis of I
  double xV0[D] = {}, xVl[D] = {};          // first / last coordinate of each axis of v
  std::vector<double> xJ[D];                // all coordinates of each axis of J
  const double* I = nullptr;                // (CI, nI...)
  const double* J = nullptr;                // (CJ, nJ...)
  const double* v = nullptr;                // (nt, nV..., D)
  const double* L = nullptr;                // (D, D)
  const double* T = nullptr;                // (D,)
  const double* K = nullptr;                // (nV...) Fourier-domain smoothing kernel, 1/LL
  const double* LL = nullptr;               // (nV...) Fourier symbol of the regulariser
  const double* WM = nullptr;               // (nJ...) initial matching / artifact /
  const double* WA = nullptr;               //         background weights
  const double* WB = nullptr;
  const double* muA = nullptr;              // (CJ,) fixed artifact / background means;
  const double* muB = nullptr;              // nullptr: re-estimated every 5 iterations
                                            // (mixture EM: the initial means, required)
  i64 NP = 0;                               // number of landmark pairs
  const double* pointsI = nullptr;          // (NP, D) landmarks in I and J
  const double* pointsJ = nullptr;
  Params params;
};

template <int D>
class Engine {
 public:
  // ---- configuration
  int nt = 0, CI = 0, CJ = 0, K1 = 0;
  int nthreads = 1;
  Grid<D> gI, gV;
  i64 m[D] = {};  // shape of J
  i64 NJ = 0, nrows = 0, rowlen = 0, NI = 0, NV = 0;
  std::vector<double> xJ[D];
  const double* I = nullptr;
  const double* J = nullptr;
  Params P;
  double cM = 1, cA = 1, cB = 1;  // Gaussian normalisers sqrt(2 pi sigma^2)^CJ
  bool est_muA = true, est_muB = true;
  i64 rows_per_chunk = 1, nchunks = 1;

  // ---- state
  int it = 0;                     // iterations done (the global iteration counter)
  std::vector<double> v;          // (nt, nV..., D), current
  double L[D][D], T[D];           // current affine
  std::vector<double> WM, WA, WB, muA, muB;
  // the last evaluated iterate: A and v used by the latest forward pass
  double A_eval[D + 1][D + 1];
  std::vector<double> v_eval;
  // affine step control (if P.affine_step_control): multipliers and previous gradient of
  // the affine parameters, L row-major then T
  static constexpr int NA = D * D + D;
  double step_m[NA], step_gprev[NA];

  // ---- outputs of the latest forward pass
  std::vector<double> Xbuf;    // (NJ, nt+1, D) sample points X^(t); X^(0) is Xs
  std::vector<double> AIbuf;   // (NJ, CI) deformed image
  std::vector<double> coeffs;  // (K1, CJ) contrast coefficients
  std::vector<double> pIt;     // (NP, D) transformed landmarks
  double gL[D][D], gT[D];      // affine gradient
  double E_last[4] = {0, 0, 0, 0};  // EM + ER, EM, ER, EP
  std::vector<double> Ehist;        // E_last of every iteration
  i64 lm_unconverged = 0;           // exact landmark solves that did not converge or fell
                                    // back to the Euler step (singular Newton matrix)

  // ---- mixture EM state: priors (matching, artifact, background) and class variances
  // (set at the first E-step)
  double mix_pi[3] = {1.0, 0.0, 0.0};
  double mix_var[3] = {0.0, 0.0, 0.0};
  bool mix_var_set = false;

  Engine(const Inputs<D>& in, int num_threads) {
    if (in.nt < 1 || in.nt > MAX_NT) throw std::invalid_argument("nt out of range");
    if (in.CI < 1 || in.CI > MAX_CI || in.CJ < 1 || in.CJ > MAX_CJ)
      throw std::invalid_argument("number of channels out of range");
    nthreads = std::max(1, num_threads);
    nt = in.nt;
    CI = in.CI;
    CJ = in.CJ;
    K1 = 1 + CI;
    P = in.params;
    gI.init(in.nI, in.xI0, in.xIl, "the image grid");
    gV.init(in.nV, in.xV0, in.xVl, "the velocity grid");
    NI = gI.npts;
    NV = gV.npts;
    I = in.I;
    J = in.J;
    NJ = 1;
    for (int d = 0; d < D; ++d) {
      m[d] = in.nJ[d];
      NJ *= m[d];
      xJ[d] = in.xJ[d];
      if ((i64)xJ[d].size() != m[d]) throw std::invalid_argument("xJ length mismatch");
    }
    rowlen = m[D - 1];
    nrows = NJ / rowlen;
    // same expression as the torch code: np.sqrt(2.0*np.pi*sigma**2)**CJ
    const double pi = 3.14159265358979323846;
    cM = std::pow(std::sqrt(2.0 * pi * (P.sigmaM * P.sigmaM)), (double)CJ);
    cA = std::pow(std::sqrt(2.0 * pi * (P.sigmaA * P.sigmaA)), (double)CJ);
    cB = std::pow(std::sqrt(2.0 * pi * (P.sigmaB * P.sigmaB)), (double)CJ);

    v.assign(in.v, in.v + nt * NV * D);
    v_eval = v;
    for (int a = 0; a < D; ++a) {
      for (int b = 0; b < D; ++b) L[a][b] = in.L[a * D + b];
      T[a] = in.T[a];
    }
    for (int a = 0; a <= D; ++a)
      for (int b = 0; b <= D; ++b)
        A_eval[a][b] = (a < D) ? (b < D ? L[a][b] : T[a]) : (b == D ? 1.0 : 0.0);
    for (int k = 0; k < NA; ++k) {
      step_m[k] = 1.0;
      step_gprev[k] = 0.0;
    }
    WM.assign(in.WM, in.WM + NJ);
    WA.assign(in.WA, in.WA + NJ);
    WB.assign(in.WB, in.WB + NJ);
    est_muA = (in.muA == nullptr);
    est_muB = (in.muB == nullptr);
    muA.assign(CJ, 0.0);
    muB.assign(CJ, 0.0);
    if (in.muA) muA.assign(in.muA, in.muA + CJ);
    if (in.muB) muB.assign(in.muB, in.muB + CJ);
    NP = in.NP;
    if (NP > 0) {
      pI.assign(in.pointsI, in.pointsI + NP * D);
      pJ.assign(in.pointsJ, in.pointsJ + NP * D);
    }
    pIt.assign(NP * D, 0.0);
    // exact landmark map: tolerance of the fixed-point solves (as the torch engine computes it)
    double cell = std::numeric_limits<double>::infinity(), extent = 0.0;
    for (int d = 0; d < D; ++d) {
      cell = std::min(cell, (in.xVl[d] - in.xV0[d]) / double(in.nV[d] - 1));
      extent = std::max(extent, std::max(std::fabs(in.xV0[d]), std::fabs(in.xVl[d])));
    }
    lm_tol = std::max(LANDMARK_TOL * cell, 64.0 * std::numeric_limits<double>::epsilon() * extent);
    if (P.mixture) {
      if (!in.muA || !in.muB) throw std::invalid_argument("the mixture EM needs the initial class means");
      if (P.mix_start < 0 || P.mix_every < 1) throw std::invalid_argument("mixture EM: need start >= 0, every >= 1");
      const double* pi = P.mix_pi_init;
      if (!(pi[0] > 0.0 && pi[1] > 0.0 && pi[2] > 0.0) || !(P.mix_pi_min >= 0.0 && P.mix_pi_min < 1.0))
        throw std::invalid_argument("mixture EM: need positive initial priors and 0 <= pi_min < 1");
      const double s = pi[0] + pi[1] + pi[2];
      for (int k = 0; k < 3; ++k) mix_pi[k] = pi[k] / s;
    }

    // spectral setup (fields over the velocity grid)
    const i64 n2 = (D == 3) ? gV.n[2] : 1;
    fft.init(gV.n[0], gV.n[1], n2, D, nt);
    const i64 h1 = fft.h1;
    const i64 SF = fft.spec_field();
    Khalf.resize(SF);
    LLw.resize(SF);
    for (i64 k0 = 0; k0 < gV.n[0]; ++k0)
      for (i64 k1 = 0; k1 < h1; ++k1)
        for (i64 k2 = 0; k2 < n2; ++k2) {
          const i64 src = (k0 * gV.n[1] + k1) * n2 + k2;
          const i64 dst = (k0 * h1 + k1) * n2 + k2;
          Khalf[dst] = in.K[src];
          // weight of a half-spectrum entry in the full-spectrum sum (LL is even)
          const bool self = (k1 == 0) || (gV.n[1] % 2 == 0 && k1 == gV.n[1] / 2);
          LLw[dst] = in.LL[src] * (self ? 1.0 : 2.0);
        }
    specA.assign(fft.spec_size(), cplx(0, 0));
    specB.assign(fft.spec_size(), cplx(0, 0));
    specA_valid = false;

    // ~256 chunks of target rows: the unit of work and of every reduction (results depend on
    // this partition at roundoff level, never on the number of threads)
    rows_per_chunk = std::max<i64>(1, (nrows + 255) / 256);
    nchunks = (nrows + rows_per_chunk - 1) / rows_per_chunk;
    NRED1 = K1 * K1 + K1 * CJ + 3 + 2 * CJ;
    NRED2 = 1 + D * (D + 1);
    red1.assign(nchunks * NRED1, 0.0);
    red2.assign(nchunks * NRED2, 0.0);
    boxes.assign(nchunks * nt, Box());
    sbufs.assign(nchunks * nt, std::vector<double>());
    chunk_max.assign(nchunks, 0.0);

    Xbuf.assign(NJ * (nt + 1) * D, 0.0);
    AIbuf.assign(NJ * CI, 0.0);
    dAIbuf.assign(NJ * CI * D, 0.0);
    rsbuf.assign(NJ, 0.0);
    gradV.assign(nt * NV * D, 0.0);
    coeffs.assign(K1 * CJ, 0.0);
  }

  // Run n iterations of the original loop body.  The iteration counter persists
  // across calls, so splitting a run into segments does not change the result.
  void run(int n) {
    if (n <= 0) return;
    acquire_pool();
    for (int k = 0; k < n; ++k) step(k == n - 1);
  }

  // Gradients at the current state, without updating it (a testing hook).  Fills
  // gL, gT and E_last; vgrad = d(EM + EP)/dv (the matching and landmark terms) and
  // direction = K * dE/dv including ER, the direction the v update subtracts.
  void gradient(std::vector<double>& vgrad, std::vector<double>& direction) {
    acquire_pool();
    double A[D + 1][D + 1], Ai[D + 1][D + 1];
    forward_backward(true, false, false, A, Ai);
    vgrad = gradV;
    // ER from fftn(v)
    fft.forward(pool(), v.data(), specB.data());
    const i64 SF = fft.spec_field();
    double er = 0.0;
    for (i64 i = 0; i < fft.spec_size(); ++i) er += LLw[(i / D) % SF] * std::norm(specB[i]);
    E_last[2] = er * er_factor();
    E_last[0] = E_last[1] + E_last[2];
    // direction = ifftn(K fftn(vgrad)) + DV/sigmaR^2 * v
    fft.forward(pool(), gradV.data(), specB.data());
    for (i64 i = 0; i < fft.spec_size(); ++i) specB[i] *= Khalf[(i / D) % SF];
    direction.resize(gradV.size());
    fft.inverse(pool(), specB.data(), direction.data(), 1.0 / double(NV));
    const double cR = P.DV / (P.sigmaR * P.sigmaR);
    for (size_t i = 0; i < direction.size(); ++i) direction[i] += cR * v[i];
  }

 private:
  static constexpr int BLK = 64;  // pixels per block along a target row

  // ---- points
  i64 NP = 0;
  std::vector<double> pI, pJ;
  double lm_tol = 0.0;  // stopping tolerance of the exact landmark map

  // ---- spectral
  FieldFFT fft;
  std::vector<double> Khalf, LLw;  // K, and LL times the half-spectrum weights
  std::vector<cplx> specA;  // fftn(v), kept up to date incrementally (for ER)
  std::vector<cplx> specB;  // work spectrum
  bool specA_valid = false;
  static constexpr int kResyncEvery = 50;  // recompute specA from v every 50 iterations

  // ---- scratch
  std::vector<double> dAIbuf, rsbuf, gradV, mu_sums, mix_red;
  i64 NRED1 = 0, NRED2 = 0;
  std::vector<double> red1, red2;
  struct Box {
    i64 lo[D], hi[D], lstr[D];
    i64 size;  // number of points (times D doubles)
    bool valid;
  };
  std::vector<Box> boxes;
  std::vector<std::vector<double>> sbufs;
  std::vector<double> chunk_max;

  ThreadPool* pool_ = nullptr;
  ThreadPool& pool() { return *pool_; }
  // The process-wide pool (re-created after fork), sized for this engine.
  void acquire_pool() {
    pool_ = &global_pool();
    pool_->resize(nthreads);
  }

  // ER = er_factor() * sum_k LLw_k |fftn(v)_k|^2 over the half spectrum
  double er_factor() const {
    return P.DV / 2.0 / double(NV) / (P.sigmaR * P.sigmaR);
  }

  // ------------------------------------------------------------------ pass 1
  // Pixels are processed in blocks of BLK along a target row with the time loop
  // outside the pixel loop, so the nt-step dependency chain of one pixel overlaps
  // with the other pixels of the block (latency bound otherwise).
  void pass1(const double (&Ai)[D][D + 1], bool mu_step) {
    pool().parallel_for(nchunks, [&](i64 ch) {
      double* r = &red1[ch * NRED1];
      // local accumulators (not aliased with the image / buffers -> kept in registers / L1)
      double bb[(MAX_CI + 1) * (MAX_CI + 1)];
      double bj[(MAX_CI + 1) * MAX_CJ];
      double mu[3 + 2 * MAX_CJ];
      for (int k = 0; k < K1 * K1; ++k) bb[k] = 0.0;
      for (int k = 0; k < K1 * CJ; ++k) bj[k] = 0.0;
      for (int k = 0; k < 3 + 2 * CJ; ++k) mu[k] = 0.0;
      i64 lo[MAX_NT][D], hi[MAX_NT][D];
      for (int t = 0; t < nt; ++t)
        for (int d = 0; d < D; ++d) {
          lo[t][d] = std::numeric_limits<i64>::max();
          hi[t][d] = -1;
        }
      const double dnt = double(nt);
      // x / nt == x * (1/nt) exactly when nt is a power of two
      const bool pow2 = (nt & (nt - 1)) == 0;
      const double inv_nt = 1.0 / dnt;
      const i64 XS = (nt + 1) * D;
      const double* __restrict xl = xJ[D - 1].data();
      const i64 r0 = ch * rows_per_chunk, r1 = std::min(nrows, r0 + rows_per_chunk);
      double x[D];
      double B[MAX_CI + 1];
      for (i64 row = r0; row < r1; ++row) {
        i64 rr = row;
        for (int d = D - 2; d >= 0; --d) {
          x[d] = xJ[d][rr % m[d]];
          rr /= m[d];
        }
        // the row-constant part of Ai x + b
        double c0[D];
        for (int d = 0; d < D; ++d) {
          double s = 0.0;
          for (int e = 0; e < D - 1; ++e) s += Ai[d][e] * x[e];
          c0[d] = s;
        }
        for (i64 b0 = 0; b0 < rowlen; b0 += BLK) {
          const i64 nb = std::min<i64>(BLK, rowlen - b0);
          const i64 j0 = row * rowlen + b0;
          double* __restrict Xb = &Xbuf[j0 * XS];
          // X^(nt) = Ai x + b (same association as torch: (Ai[:, :D] @ x) + Ai[:, D])
          for (i64 b = 0; b < nb; ++b) {
            const double xe = xl[b0 + b];
            for (int d = 0; d < D; ++d) {
              double s;
              if (D == 2) s = Ai[d][0] * x[0] + Ai[d][1] * xe;
              else s = c0[d] + Ai[d][D - 1] * xe;
              Xb[b * XS + nt * D + d] = s + Ai[d][D];
            }
          }
          for (int t = nt - 1; t >= 0; --t) {
            const double* vt = &v[t * NV * D];
            for (i64 b = 0; b < nb; ++b) {
              const double* Xp = &Xb[b * XS + (t + 1) * D];
              double* Xn = &Xb[b * XS + t * D];
              VSample<D> s;
              if (vsetup(gV, Xp, s)) {
                double val[D];
                vvalue(gV, vt, s, val);
                for (int d = 0; d < D; ++d) {
                  Xn[d] = Xp[d] - (pow2 ? val[d] * inv_nt : val[d] / dnt);
                  const i64 l = s.ok0[d] ? s.i0[d] : s.i0[d] + 1;
                  const i64 h = s.ok1[d] ? s.i0[d] + 1 : s.i0[d];
                  lo[t][d] = std::min(lo[t][d], l);
                  hi[t][d] = std::max(hi[t][d], h);
                }
              } else {
                for (int d = 0; d < D; ++d) Xn[d] = Xp[d];
              }
            }
          }
          for (i64 b = 0; b < nb; ++b) {
            const i64 j = j0 + b;
            double* ai = &AIbuf[j * CI];
            isample<D>(gI, I, NI, CI, &Xb[b * XS], ai, &dAIbuf[j * CI * D]);
            const double wm = WM[j];
            B[0] = 1.0;
            for (int i = 0; i < CI; ++i) B[1 + i] = ai[i];
            for (int k = 0; k < K1; ++k) {
              for (int l = k; l < K1; ++l) bb[k * K1 + l] += B[k] * (B[l] * wm);
              for (int c = 0; c < CJ; ++c) bj[k * CJ + c] += B[k] * (J[c * NJ + j] * wm);
            }
            if (mu_step) {
              const double wa = WA[j], wb = WB[j];
              mu[0] += wm;
              mu[1] += wa;
              mu[2] += wb;
              for (int c = 0; c < CJ; ++c) {
                mu[3 + c] += wa * J[c * NJ + j];
                mu[3 + CJ + c] += wb * J[c * NJ + j];
              }
            }
          }
        }
      }
      std::copy(bb, bb + K1 * K1, r);
      std::copy(bj, bj + K1 * CJ, r + K1 * K1);
      std::copy(mu, mu + 3 + 2 * CJ, r + K1 * K1 + K1 * CJ);
      for (int t = 0; t < nt; ++t) {
        Box& bx = boxes[ch * nt + t];
        bx.valid = hi[t][0] >= 0;
        i64 sz = 1;
        for (int d = D - 1; d >= 0; --d) {
          bx.lo[d] = bx.valid ? lo[t][d] : 0;
          bx.hi[d] = bx.valid ? hi[t][d] : -1;
          bx.lstr[d] = sz;
          sz *= bx.valid ? (bx.hi[d] - bx.lo[d] + 1) : 0;
        }
        bx.size = bx.valid ? sz : 0;
      }
    });
  }

  // ------------------------------------------------------------------ pass 2
  void pass2(bool splat, bool e_step) {
    const double invsM2 = 1.0 / (P.sigmaM * P.sigmaM);
    const double dnt = double(nt);
    const bool pow2 = (nt & (nt - 1)) == 0;
    const double inv_nt = 1.0 / dnt;
    pool().parallel_for(nchunks, [&](i64 ch) {
      double* r = &red2[ch * NRED2];
      double gAi[D * (D + 1)];
      for (int k = 0; k < D * (D + 1); ++k) gAi[k] = 0.0;
      double em = 0.0;
      if (splat)
        for (int t = 0; t < nt; ++t) {
          auto& sb = sbufs[ch * nt + t];
          const Box& b = boxes[ch * nt + t];
          sb.assign(b.size * D, 0.0);
        }
      const i64 XS = (nt + 1) * D;
      const double* __restrict xl = xJ[D - 1].data();
      const double* __restrict cf = coeffs.data();
      const i64 r0 = ch * rows_per_chunk, r1 = std::min(nrows, r0 + rows_per_chunk);
      double x[D];
      double dfa[MAX_CJ];
      double G[BLK][D];
      for (i64 row = r0; row < r1; ++row) {
        i64 rr = row;
        for (int d = D - 2; d >= 0; --d) {
          x[d] = xJ[d][rr % m[d]];
          rr /= m[d];
        }
        for (i64 b0 = 0; b0 < rowlen; b0 += BLK) {
          const i64 nb = std::min<i64>(BLK, rowlen - b0);
          const i64 j0 = row * rowlen + b0;
          const double* __restrict Xb = &Xbuf[j0 * XS];
          // dE/dAI and its image-gradient projection g^(0)
          for (i64 b = 0; b < nb; ++b) {
            const i64 j = j0 + b;
            const double* ai = &AIbuf[j * CI];
            const double* dai = &dAIbuf[j * CI * D];
            const double wm = WM[j];
            double rs = 0.0;
            for (int c = 0; c < CJ; ++c) {
              double f = cf[c];
              for (int i = 0; i < CI; ++i) f += ai[i] * cf[(1 + i) * CJ + c];
              const double res = f - J[c * NJ + j];
              const double r2 = res * res;
              rs += r2;
              em += r2 * wm;
              dfa[c] = (invsM2 * wm) * res;
            }
            double g[D];
            for (int d = 0; d < D; ++d) g[d] = 0.0;
            for (int i = 0; i < CI; ++i) {
              double s = 0.0;
              for (int c = 0; c < CJ; ++c) s += cf[(1 + i) * CJ + c] * dfa[c];
              for (int d = 0; d < D; ++d) g[d] += s * dai[i * D + d];
            }
            for (int d = 0; d < D; ++d) G[b][d] = g[d];
            if (e_step) rsbuf[j] = rs;
          }
          // backprop through the velocity steps t = 0 .. nt-1
          for (int t = 0; t < nt; ++t) {
            const double* vt = &v[t * NV * D];
            if (splat) {
              const Box& bx = boxes[ch * nt + t];
              double* lb = sbufs[ch * nt + t].data();
              SplatAcc<D> sa;
              for (i64 b = 0; b < nb; ++b) {
                VSample<D> s;
                if (!vsetup(gV, &Xb[b * XS + (t + 1) * D], s)) continue;
                double q[D], JTq[D];
                for (int d = 0; d < D; ++d) q[d] = pow2 ? G[b][d] * inv_nt : G[b][d] / dnt;
                vback_acc(gV, vt, s, q, lb, bx.lo, bx.lstr, JTq, sa);
                for (int d = 0; d < D; ++d) G[b][d] -= JTq[d];
              }
              sa.flush(lb, bx.lstr);
            } else {
              for (i64 b = 0; b < nb; ++b) {
                VSample<D> s;
                if (!vsetup(gV, &Xb[b * XS + (t + 1) * D], s)) continue;
                double q[D], JTq[D];
                for (int d = 0; d < D; ++d) q[d] = pow2 ? G[b][d] * inv_nt : G[b][d] / dnt;
                vback<D, false>(gV, vt, s, q, -1.0, nullptr, nullptr, nullptr, JTq);
                for (int d = 0; d < D; ++d) G[b][d] -= JTq[d];
              }
            }
          }
          for (i64 b = 0; b < nb; ++b) {
            x[D - 1] = xl[b0 + b];
            for (int d = 0; d < D; ++d) {
              for (int e = 0; e < D; ++e) gAi[d * (D + 1) + e] += G[b][d] * x[e];
              gAi[d * (D + 1) + D] += G[b][d];
            }
          }
        }
      }
      r[0] = em;
      std::copy(gAi, gAi + D * (D + 1), r + 1);
    });
  }

  // reduce chunk splat boxes into gradV (deterministic: chunk order per element)
  void reduce_splats() {
    const i64 n0 = gV.n[0];
    const i64 nrow = nt * n0;
    const i64 ntask = std::min<i64>(nrow, 48);
    pool().parallel_for(ntask, [&](i64 tk) {
      for (i64 task = nrow * tk / ntask; task < nrow * (tk + 1) / ntask; ++task) {
        const int t = (int)(task / n0);
        const i64 i0 = task % n0;
        double* G = &gradV[(t * NV + i0 * gV.str[0]) * D];
        std::fill(G, G + gV.str[0] * D, 0.0);
        for (i64 ch = 0; ch < nchunks; ++ch) {
          const Box& b = boxes[ch * nt + t];
          if (!b.valid || i0 < b.lo[0] || i0 > b.hi[0]) continue;
          const double* src = sbufs[ch * nt + t].data() + (i0 - b.lo[0]) * b.lstr[0] * D;
          if (D == 2) {
            double* dst = G + b.lo[1] * D;
            const i64 len = (b.hi[1] - b.lo[1] + 1) * D;
            for (i64 k = 0; k < len; ++k) dst[k] += src[k];
          } else {
            const i64 len = (b.hi[D - 1] - b.lo[D - 1] + 1) * D;
            for (i64 i1 = b.lo[1]; i1 <= b.hi[1]; ++i1) {
              double* dst = G + (i1 * gV.str[1] + b.lo[D - 1]) * D;
              const double* s2 = src + (i1 - b.lo[1]) * b.lstr[1] * D;
              for (i64 k = 0; k < len; ++k) dst[k] += s2[k];
            }
          }
        }
      }
    });
  }

  // Landmark term: forward + backward; adds its splats into gradV when `splat`.
  // Points move forward in time for t = 0..nt-1, then by A: by Euler steps
  // p <- p + v_t(p)/nt, or (P.landmark_exact) by the implicit steps p <- z with
  // z = p + v_t(z)/nt, solved by fixed-point iteration from the Euler step and finished
  // by one Newton step z + M^-1 (p + v_t(z)/nt - z), M = I - Dv_t(z)/nt (the solution
  // up to rounding, whatever the stopping tolerance).  The gradient uses the same z and M.
  // Where M is exactly singular (det M = 0, or NaN) that step is the Euler step instead,
  // with its gradient, and is counted in lm_unconverged (as the torch engine does).
  double points_term(const double (&A)[D + 1][D + 1], double (&gA)[D][D + 1], bool splat) {
    const double dnt = double(nt);
    const double invsP2 = 1.0 / (P.sigmaP * P.sigmaP);
    double EP = 0.0;
    std::vector<double> Pt((nt + 1) * D);
    // exact map: the fixed point z of each step before the Newton step, its M, whether v_t
    // has a sample there, and whether the step fell back to the Euler step
    std::vector<double> Zt(nt * D), Mt(nt * D * D);
    std::vector<char> Zin(nt, 0), Zeuler(nt, 0);
    for (i64 ip = 0; ip < NP; ++ip) {
      for (int d = 0; d < D; ++d) Pt[d] = pI[ip * D + d];
      for (int t = 0; t < nt; ++t) {
        const double* Pc = &Pt[t * D];
        double* Pn = &Pt[(t + 1) * D];
        const double* vt = &v[t * NV * D];
        VSample<D> s;
        double val[D] = {};
        if (vsetup(gV, Pc, s)) vvalue(gV, vt, s, val);
        for (int d = 0; d < D; ++d) Pn[d] = Pc[d] + val[d] / dnt;
        if (!P.landmark_exact) continue;
        double euler[D];
        std::copy(Pn, Pn + D, euler);
        Zeuler[t] = 0;
        bool converged = false;
        for (int k = 0; k < LANDMARK_MAX_ITER && !converged; ++k) {
          double w[D] = {};
          if (vsetup(gV, Pn, s)) vvalue(gV, vt, s, w);
          double moved = 0.0;
          for (int d = 0; d < D; ++d) {
            const double z = Pc[d] + w[d] / dnt;
            moved = std::max(moved, std::fabs(z - Pn[d]));
            Pn[d] = z;
          }
          converged = !(moved > lm_tol);
        }
        if (!converged) ++lm_unconverged;
        std::copy(Pn, Pn + D, &Zt[t * D]);
        Zin[t] = vsetup(gV, Pn, s);
        if (!Zin[t]) continue;  // v_t = 0 around z: z = Pc exactly
        double w[D], Jac[D][D], M[D * D], r[D];
        vvalue(gV, vt, s, w);
        vjacobian(gV, vt, s, Jac);
        double* Ms = &Mt[t * D * D];
        for (int a = 0; a < D; ++a) {
          for (int b = 0; b < D; ++b) Ms[a * D + b] = (a == b ? 1.0 : 0.0) - Jac[a][b] / dnt;
          r[a] = Pc[a] + w[a] / dnt - Pn[a];
        }
        if (!(std::fabs(small_det(D, Ms)) > 0.0)) {  // singular M: the Euler step
          if (converged) ++lm_unconverged;
          Zeuler[t] = 1;
          std::copy(euler, euler + D, Pn);
          continue;
        }
        std::copy(Ms, Ms + D * D, M);
        small_solve(D, 1, M, r);
        for (int d = 0; d < D; ++d) Pn[d] += r[d];
      }
      const double* Pf = &Pt[nt * D];
      double pf[D], dp[D];
      for (int d = 0; d < D; ++d) {
        double s = A[d][0] * Pf[0];
        for (int e = 1; e < D; ++e) s += A[d][e] * Pf[e];
        pf[d] = s + A[d][D];
        pIt[ip * D + d] = pf[d];
        const double r = pf[d] - pJ[ip * D + d];
        EP += r * r;
        dp[d] = invsP2 * r;
      }
      double g[D];
      for (int d = 0; d < D; ++d) {
        for (int e = 0; e < D; ++e) gA[d][e] += dp[d] * Pf[e];
        gA[d][D] += dp[d];
      }
      for (int e = 0; e < D; ++e) {
        double s = 0.0;
        for (int d = 0; d < D; ++d) s += A[d][e] * dp[d];
        g[e] = s;
      }
      i64 lo[D], lstr[D];  // splat straight into gradV
      for (int d = 0; d < D; ++d) {
        lo[d] = 0;
        lstr[d] = gV.str[d];
      }
      for (int t = nt - 1; t >= 0; --t) {
        const double* vt = &v[t * NV * D];
        VSample<D> s;
        double q[D], JTq[D];
        if (P.landmark_exact && !Zeuler[t]) {
          // implicit function theorem at the solution z: lambda = M^-T g is the gradient
          // w.r.t. Pt[t], and lambda/nt is splatted at z
          if (!Zin[t]) continue;  // v_t = 0 near z: lambda = g
          vsetup(gV, &Zt[t * D], s);
          const double* Ms = &Mt[t * D * D];
          double MT[D * D], lam[D];
          for (int a = 0; a < D; ++a) {
            for (int b = 0; b < D; ++b) MT[a * D + b] = Ms[b * D + a];
            lam[a] = g[a];
          }
          small_solve(D, 1, MT, lam);
          for (int d = 0; d < D; ++d) {
            q[d] = lam[d] / dnt;
            g[d] = lam[d];
          }
          if (splat) vback<D, true>(gV, vt, s, q, 1.0, &gradV[t * NV * D], lo, lstr, JTq);
          continue;
        }
        if (!vsetup(gV, &Pt[t * D], s)) continue;
        for (int d = 0; d < D; ++d) q[d] = g[d] / dnt;
        if (splat) {
          vback<D, true>(gV, vt, s, q, 1.0, &gradV[t * NV * D], lo, lstr, JTq);
        } else {
          vback<D, false>(gV, vt, s, q, 1.0, nullptr, nullptr, nullptr, JTq);
        }
        for (int d = 0; d < D; ++d) g[d] += JTq[d];
      }
    }
    return EP / 2.0 / (P.sigmaP * P.sigmaP);
  }

  // sum_k LLw |specA_k|^2 (parallel, deterministic: fixed task partition)
  double er_sum_specA() {
    const i64 SF = fft.spec_field();
    const i64 N = fft.spec_size();
    const i64 ntask = std::max<i64>(1, std::min<i64>(128, N / 8192));
    std::vector<double> part(ntask, 0.0);
    pool().parallel_for(ntask, [&](i64 tk) {
      const i64 a = N * tk / ntask, b = N * (tk + 1) / ntask;
      double acc = 0.0;
      for (i64 i = a; i < b; ++i) acc += LLw[(i / D) % SF] * std::norm(specA[i]);
      part[tk] = acc;
    });
    double tot = 0.0;
    for (double x : part) tot += x;
    return tot;
  }

  // forward + backward for the current state; fills gL, gT, E_last[1, 3], coeffs,
  // AIbuf, Xbuf, mu_sums (if mu_step), rsbuf (if e_step) and gradV (if splat)
  void forward_backward(bool splat, bool mu_step, bool e_step, double (&A)[D + 1][D + 1],
                        double (&Ai)[D + 1][D + 1]) {
    for (int a = 0; a < D; ++a) {
      for (int b = 0; b < D; ++b) A[a][b] = L[a][b];
      A[a][D] = T[a];
    }
    for (int b = 0; b < D; ++b) A[D][b] = 0.0;
    A[D][D] = 1.0;
    small_inverse(D + 1, &A[0][0], &Ai[0][0]);
    double Ai_r[D][D + 1];
    for (int a = 0; a < D; ++a)
      for (int b = 0; b <= D; ++b) Ai_r[a][b] = Ai[a][b];

    pass1(Ai_r, mu_step);
    // reduce pass-1 partials in chunk order
    std::vector<double> R1(NRED1, 0.0);
    for (i64 ch = 0; ch < nchunks; ++ch)
      for (i64 k = 0; k < NRED1; ++k) R1[k] += red1[ch * NRED1 + k];
    std::vector<double> BB(K1 * K1), BJ(K1 * CJ);
    for (int k = 0; k < K1; ++k)
      for (int l = 0; l < K1; ++l) BB[k * K1 + l] = (l >= k) ? R1[k * K1 + l] : R1[l * K1 + k];
    for (int k = 0; k < K1; ++k) BB[k * K1 + k] += 0.1;
    for (int k = 0; k < K1 * CJ; ++k) BJ[k] = R1[K1 * K1 + k];
    small_solve(K1, CJ, BB.data(), BJ.data());
    coeffs = BJ;
    if (mu_step) {
      const double* MU = &R1[K1 * K1 + K1 * CJ];
      mu_sums.assign(MU, MU + 3 + 2 * CJ);
    }

    pass2(splat, e_step);
    std::vector<double> R2(NRED2, 0.0);
    for (i64 ch = 0; ch < nchunks; ++ch)
      for (i64 k = 0; k < NRED2; ++k) R2[k] += red2[ch * NRED2 + k];
    const double EM = R2[0] / 2.0 / (P.sigmaM * P.sigmaM);
    double G[D + 1][D + 1] = {};
    for (int a = 0; a < D; ++a)
      for (int b = 0; b <= D; ++b) G[a][b] = R2[1 + a * (D + 1) + b];
    // dE/dA = -Ai^T G Ai^T
    double tmp[D + 1][D + 1] = {}, gA[D + 1][D + 1] = {};
    for (int a = 0; a <= D; ++a)
      for (int b = 0; b <= D; ++b) {
        double s = 0.0;
        for (int k = 0; k <= D; ++k) s += Ai[k][a] * G[k][b];
        tmp[a][b] = s;
      }
    for (int a = 0; a <= D; ++a)
      for (int b = 0; b <= D; ++b) {
        double s = 0.0;
        for (int k = 0; k <= D; ++k) s += tmp[a][k] * Ai[b][k];
        gA[a][b] = -s;
      }
    if (splat) reduce_splats();
    double gAp[D][D + 1] = {};
    double EP = 0.0;
    if (NP > 0) EP = points_term(A, gAp, splat);
    for (int a = 0; a < D; ++a) {
      for (int b = 0; b < D; ++b) gL[a][b] = gA[a][b] + gAp[a][b];
      gT[a] = gA[a][D] + gAp[a][D];
    }
    E_last[1] = EM;
    E_last[3] = EP;
  }

  // E-step of the weights from this iteration's residuals (rsbuf); muA/muB have
  // already been updated by the M-step.
  void estep() {
    const double* MU = mu_sums.data();
    double pi[3] = {MU[0], MU[1], MU[2]};
    const double mx = std::max(pi[0], std::max(pi[1], pi[2]));
    for (double& p : pi) p += mx * 1e-6;
    const double sp = pi[0] + pi[1] + pi[2];
    for (double& p : pi) p /= sp;
    const double sM2 = P.sigmaM * P.sigmaM, sA2 = P.sigmaA * P.sigmaA, sB2 = P.sigmaB * P.sigmaB;
    pool().parallel_for(nchunks, [&](i64 ch) {
      const i64 j0 = ch * rows_per_chunk * rowlen, j1 = std::min(NJ, j0 + rows_per_chunk * rowlen);
      double cm = -std::numeric_limits<double>::infinity();
      for (i64 j = j0; j < j1; ++j) {
        double sa = 0.0, sb = 0.0;
        for (int c = 0; c < CJ; ++c) {
          const double da = muA[c] - J[c * NJ + j];
          const double db = muB[c] - J[c * NJ + j];
          sa += da * da;
          sb += db * db;
        }
        const double wm = pi[0] * std::exp(-rsbuf[j] / 2.0 / sM2) / cM;
        const double wa = pi[1] * std::exp(-sa / 2.0 / sA2) / cA;
        const double wb = pi[2] * std::exp(-sb / 2.0 / sB2) / cB;
        WM[j] = wm;
        WA[j] = wa;
        WB[j] = wb;
        const double ws = wm + wb + wa;
        if (ws > cm) cm = ws;
      }
      chunk_max[ch] = cm;
    });
    double mxs = -std::numeric_limits<double>::infinity();
    for (i64 ch = 0; ch < nchunks; ++ch) mxs = std::max(mxs, chunk_max[ch]);
    const double add = mxs * 1e-6;
    pool().parallel_for(nchunks, [&](i64 ch) {
      const i64 j0 = ch * rows_per_chunk * rowlen, j1 = std::min(NJ, j0 + rows_per_chunk * rowlen);
      for (i64 j = j0; j < j1; ++j) {
        const double ws = (WM[j] + WB[j] + WA[j]) + add;
        WM[j] /= ws;
        WB[j] /= ws;
        WA[j] /= ws;
      }
    });
  }

  // Mixture EM (P.mixture): E-step from this iteration's residuals (rsbuf), then the M-step
  // of the priors, the estimated class means and the variances.  The per-pixel work and its
  // sums run per chunk and are combined in chunk order.
  void mixture_em() {
    const double C = double(CJ);
    const i64 rows = rows_per_chunk * rowlen;
    if (!mix_var_set) {  // first E-step: the variance of the unweighted residual
      std::vector<double> part(nchunks, 0.0);
      pool().parallel_for(nchunks, [&](i64 ch) {
        double s = 0.0;
        for (i64 j = ch * rows, j1 = std::min(NJ, j + rows); j < j1; ++j) s += rsbuf[j];
        part[ch] = s;
      });
      double tot = 0.0;
      for (double x : part) tot += x;
      mix_var[0] = std::max(tot / double(NJ) / C, VARIANCE_FLOOR);
      set_class_variances();
      mix_var_set = true;
    }
    const bool on[3] = {true, P.mix_classA, P.mix_classB};
    double lp[3], hl[3];
    for (int k = 0; k < 3; ++k) {
      lp[k] = std::log(mix_pi[k]);
      hl[k] = 0.5 * C * std::log(mix_var[k]);
    }
    const double vM2 = 2.0 * mix_var[0], vA2 = 2.0 * mix_var[1], vB2 = 2.0 * mix_var[2];
    const double ninf = -std::numeric_limits<double>::infinity();
    // per chunk: sum WM, sum WA, sum WB, sum WM |r|^2, sum WA J (CJ), sum WB J (CJ)
    const i64 NS = 4 + 2 * CJ;
    mix_red.assign(nchunks * NS, 0.0);
    pool().parallel_for(nchunks, [&](i64 ch) {
      double acc[4 + 2 * MAX_CJ];
      for (i64 k = 0; k < NS; ++k) acc[k] = 0.0;
      for (i64 j = ch * rows, j1 = std::min(NJ, j + rows); j < j1; ++j) {
        double d2a = 0.0, d2b = 0.0;
        for (int c = 0; c < CJ; ++c) {
          const double da = J[c * NJ + j] - muA[c];
          const double db = J[c * NJ + j] - muB[c];
          d2a += da * da;
          d2b += db * db;
        }
        const double l0 = (lp[0] - rsbuf[j] / vM2) - hl[0];
        const double l1 = on[1] ? (lp[1] - d2a / vA2) - hl[1] : ninf;
        const double l2 = on[2] ? (lp[2] - d2b / vB2) - hl[2] : ninf;
        const double mx = std::max(l0, std::max(l1, l2));
        const double e0 = std::exp(l0 - mx), e1 = on[1] ? std::exp(l1 - mx) : 0.0,
                     e2 = on[2] ? std::exp(l2 - mx) : 0.0;
        const double s = e0 + e1 + e2;
        const double wm = e0 / s, wa = e1 / s, wb = e2 / s;
        WM[j] = wm;
        WA[j] = wa;
        WB[j] = wb;
        acc[0] += wm;
        acc[1] += wa;
        acc[2] += wb;
        acc[3] += wm * rsbuf[j];
        for (int c = 0; c < CJ; ++c) {
          acc[4 + c] += wa * J[c * NJ + j];
          acc[4 + CJ + c] += wb * J[c * NJ + j];
        }
      }
      std::copy(acc, acc + NS, &mix_red[ch * NS]);
    });
    std::vector<double> S(NS, 0.0);
    for (i64 ch = 0; ch < nchunks; ++ch)
      for (i64 k = 0; k < NS; ++k) S[k] += mix_red[ch * NS + k];
    // M-step
    const double N = S[0] + S[1] + S[2];
    double pi[3];
    for (int k = 0; k < 3; ++k) pi[k] = std::max(S[k] / N, 1e-4);
    if (pi[0] < P.mix_pi_min) {
      const double rest = pi[1] + pi[2];
      pi[1] = pi[1] / rest * (1.0 - P.mix_pi_min);
      pi[2] = pi[2] / rest * (1.0 - P.mix_pi_min);
      pi[0] = P.mix_pi_min;
    }
    const double sp = pi[0] + pi[1] + pi[2];
    for (int k = 0; k < 3; ++k) mix_pi[k] = pi[k] / sp;
    if (P.mix_est_muA && S[1] > 1e-3)
      for (int c = 0; c < CJ; ++c) muA[c] = S[4 + c] / S[1];
    if (P.mix_est_muB && S[2] > 1e-3)
      for (int c = 0; c < CJ; ++c) muB[c] = S[4 + CJ + c] / S[2];
    if (S[0] > 0.0) mix_var[0] = std::max(S[3] / (C * S[0]), VARIANCE_FLOOR);
    set_class_variances();
  }

  // artifact and background variances tied to the matching variance by sigmaA/sigmaM, sigmaB/sigmaM
  void set_class_variances() {
    const double rA = P.sigmaA / P.sigmaM, rB = P.sigmaB / P.sigmaM;
    mix_var[1] = rA * rA * mix_var[0];
    mix_var[2] = rB * rB * mix_var[0];
  }

  // Affine step control: the sign of (new gradient * previous gradient) of each affine
  // parameter shrinks (< 0) or grows (> 0) its step multiplier; 0 leaves it unchanged.
  void update_step_multipliers() {
    double g[NA];
    for (int a = 0; a < D; ++a) {
      for (int b = 0; b < D; ++b) g[a * D + b] = gL[a][b];
      g[D * D + a] = gT[a];
    }
    for (int k = 0; k < NA; ++k) {
      const double s = g[k] * step_gprev[k];
      if (s < 0.0)
        step_m[k] = std::max(step_m[k] * P.step_shrink, P.step_floor);
      else if (s > 0.0)
        step_m[k] = std::min(step_m[k] * P.step_grow, P.step_cap);
      step_gprev[k] = g[k];
    }
  }

  // One iteration, with the control flow of the original loop body.  `last`:
  // keep a copy of the v used by this iteration's forward pass (v_eval).
  void step(bool last) {
    const bool diffeo = it >= P.diffeo_start;
    // weights: the original M-step (every 5 iterations) and E-step (from iteration 50), or
    // the mixture EM; both need this iteration's squared residuals (e_step)
    const bool em = P.mixture && it >= P.mix_start && (it - P.mix_start) % P.mix_every == 0;
    const bool mu_step = !P.mixture && (it % 5) == 0;
    const bool e_step = P.mixture ? em : (mu_step && it >= 50);
    double A[D + 1][D + 1], Ai[D + 1][D + 1];
    forward_backward(diffeo, mu_step, e_step, A, Ai);
    for (int a = 0; a <= D; ++a)
      for (int b = 0; b <= D; ++b) A_eval[a][b] = A[a][b];
    if (last) v_eval = v;

    // ---- ER (at the current v) and the velocity update.  specA = fftn(v) is
    // updated incrementally with v and recomputed every kResyncEvery iterations.
    const i64 n0 = fft.n0, n1 = fft.n1, n2 = fft.n2, h1 = fft.h1;
    const double epV = P.epV;
    const double cR = P.DV / (P.sigmaR * P.sigmaR);
    if (!specA_valid || it % kResyncEvery == 0) {
      fft.forward(pool(), v.data(), specA.data());
      specA_valid = true;
    }
    double ER = 0.0;
    if (diffeo) {
      // v -= epV * (ifftn(K fftn(dE/dv without ER)) + cR v), since the smoothed ER
      // gradient is exactly cR v.  The last c2c axis is fused with the K multiply,
      // the ER sum and the specA update; the final c2r with the v update.
      fft.r2c1(pool(), gradV.data(), specB.data());
      // per spectral entry e: z = fftn(gradV), a = specA (= fftn(v) before the update)
      std::vector<double> part;
      auto filter = [&](i64 task, i64 e, cplx& a, cplx z) {
        part[task] += LLw[e] * std::norm(a);
        z *= Khalf[e];
        a -= (z + cR * a) * epV;
        return z;
      };
      if constexpr (D == 2) {
        // lines (t, k1, c) along axis 0
        const i64 PL = h1 * D;  // lines per t
        const i64 nl = nt * PL;
        const i64 gr = FieldFFT::grain(n0, nl);
        part.assign(ntasks_for(nl, gr), 0.0);
        auto idx = [=](i64 l, i64 k) { return (l / PL) * n0 * PL + k * PL + l % PL; };
        c2c_lines_filter(
            pool(), *fft.c0, n0, nl, gr,
            [&](i64, i64 l, i64 k) { return specB[idx(l, k)]; },
            [&](i64 task, i64 l, i64 k, cplx z) {
              return filter(task, k * h1 + (l % PL) / D, specA[idx(l, k)], z);
            },
            [&](i64, i64 l, i64 k, cplx z) { specB[idx(l, k)] = z; });
      } else {
        // axis 0 first, then lines (t, k0, k1, c) along axis 2
        fft.c2c0(pool(), specB.data(), true);
        const i64 nl = fft.F * n0 * h1;
        const i64 gr = FieldFFT::grain(n2, nl);
        const i64 nfl = n0 * h1;
        part.assign(ntasks_for(nl, gr), 0.0);
        auto idx = [=](i64 l, i64 k) { return (l / D) * n2 * D + k * D + l % D; };
        auto eix = [=](i64 l, i64 k) { return ((l / D) % nfl) * n2 + k; };
        c2c_lines_filter(
            pool(), *fft.c2, n2, nl, gr,
            [&](i64, i64 l, i64 k) { return specB[idx(l, k)]; },
            [&](i64 task, i64 l, i64 k, cplx z) {
              return filter(task, eix(l, k), specA[idx(l, k)], z);
            },
            [&](i64, i64 l, i64 k, cplx z) { specB[idx(l, k)] = z; });
        fft.c2c0(pool(), specB.data(), false);
      }
      for (double x : part) ER += x;
      fft.c2r1(pool(), specB.data(), v.data(), 1.0 / double(n0 * n1 * n2),
               [cR, epV](double& dst, double val) { dst -= (val + cR * dst) * epV; });
    } else {
      ER = er_sum_specA();
    }
    ER *= er_factor();
    E_last[2] = ER;
    E_last[0] = E_last[1] + ER;  // like the original's Esave: E before adding EP
    Ehist.insert(Ehist.end(), E_last, E_last + 4);

    // ---- affine update (the step is 10x smaller once the diffeomorphism is on)
    const double fac = 1.0 + (diffeo ? 9.0 : 0.0);
    const double sL = P.epL / fac, sT = P.epT / fac;
    if (P.affine_step_control) {
      update_step_multipliers();
      for (int a = 0; a < D; ++a) {
        for (int b = 0; b < D; ++b) L[a][b] -= (sL * step_m[a * D + b]) * gL[a][b];
        T[a] -= (sT * step_m[D * D + a]) * gT[a];
      }
    } else {
      for (int a = 0; a < D; ++a) {
        for (int b = 0; b < D; ++b) L[a][b] -= sL * gL[a][b];
        T[a] -= sT * gT[a];
      }
    }

    // ---- weights: M-step for muA/muB, then the E-step (from iteration 50)
    if (mu_step) {
      const double* MU = mu_sums.data();
      if (est_muA)
        for (int c = 0; c < CJ; ++c) muA[c] = MU[3 + c] / MU[1];
      if (est_muB)
        for (int c = 0; c < CJ; ++c) muB[c] = MU[3 + CJ + c] / MU[2];
      if (e_step) estep();
    }
    if (em) mixture_em();
    ++it;
  }
};

}  // namespace stalign
