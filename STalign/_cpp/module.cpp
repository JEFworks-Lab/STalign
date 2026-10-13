// pybind11 bindings of the C++ LDDMM engine: extension module STalign._stalign_cpp.
//
// Engine2D / Engine3D wrap stalign::Engine<2> / <3>.  Arrays are float64 and C
// contiguous; inputs that already are (as the Python adapter passes them) are
// used without copying.  run() and gradient() release the GIL.  Every array the
// engine returns is a fresh copy owned by the caller.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "engine.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using stalign::i64;

namespace {

using carr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using shape_t = std::vector<py::ssize_t>;

py::array_t<double> to_numpy(std::vector<double>&& data, const shape_t& shape) {
  auto* p = new std::vector<double>(std::move(data));
  py::capsule owner(p, [](void* q) { delete static_cast<std::vector<double>*>(q); });
  return py::array_t<double>(shape, p->data(), owner);
}

void check_shape(const carr& a, const shape_t& shape, const char* name) {
  bool ok = a.ndim() == (py::ssize_t)shape.size();
  for (size_t d = 0; ok && d < shape.size(); ++d) ok = a.shape(d) == shape[d];
  if (!ok) {
    std::string s = "(";
    for (size_t d = 0; d < shape.size(); ++d) s += std::to_string(shape[d]) + (d + 1 < shape.size() ? ", " : ")");
    throw std::invalid_argument(std::string(name) + " must have shape " + s);
  }
}

template <int D>
class PyEngine {
 public:
  PyEngine(carr I, const std::vector<carr>& xI, carr J, const std::vector<carr>& xJ, carr v,
           const std::vector<carr>& xv, carr L, carr T, carr K, carr LL, carr WM, carr WA,
           carr WB, std::optional<carr> muA, std::optional<carr> muB, std::optional<carr> pointsI,
           std::optional<carr> pointsJ, double sigmaM, double sigmaA, double sigmaB,
           double sigmaR, double sigmaP, double epL, double epT, double epV, int diffeo_start,
           double DV, bool affine_step_control, double step_shrink, double step_grow,
           double step_floor, double step_cap, int num_threads, bool landmark_exact, bool mixture,
           int mix_start, int mix_every, const std::string& mix_classes, double mix_pi_min,
           std::array<double, 3> mix_pi_init, bool mix_est_muA, bool mix_est_muB)
      : I_(std::move(I)), J_(std::move(J)) {
    if (I_.ndim() != D + 1) throw std::invalid_argument("I must have shape (C_I, *nI)");
    if (J_.ndim() != D + 1) throw std::invalid_argument("J must have shape (C_J, *nJ)");
    if (v.ndim() != D + 2 || v.shape(D + 1) != D)
      throw std::invalid_argument("v must have shape (nt, *nv, D)");
    if (xI.size() != D || xJ.size() != D || xv.size() != D)
      throw std::invalid_argument("xI, xJ and xv must have one coordinate array per axis");
    stalign::Inputs<D> in;
    in.nt = (int)v.shape(0);
    in.CI = (int)I_.shape(0);
    in.CJ = (int)J_.shape(0);
    shape_t sv, sj;
    for (int d = 0; d < D; ++d) {
      in.nI[d] = I_.shape(1 + d);
      in.nJ[d] = J_.shape(1 + d);
      in.nV[d] = v.shape(1 + d);
      check_shape(xI[d], {in.nI[d]}, "xI[d]");
      check_shape(xJ[d], {in.nJ[d]}, "xJ[d]");
      check_shape(xv[d], {in.nV[d]}, "xv[d]");
      in.xI0[d] = xI[d].data()[0];
      in.xIl[d] = xI[d].data()[in.nI[d] - 1];
      in.xV0[d] = xv[d].data()[0];
      in.xVl[d] = xv[d].data()[in.nV[d] - 1];
      in.xJ[d].assign(xJ[d].data(), xJ[d].data() + in.nJ[d]);
      sv.push_back(in.nV[d]);
      sj.push_back(in.nJ[d]);
    }
    check_shape(L, {D, D}, "L");
    check_shape(T, {D}, "T");
    check_shape(K, sv, "K");
    check_shape(LL, sv, "LL");
    check_shape(WM, sj, "WM");
    check_shape(WA, sj, "WA");
    check_shape(WB, sj, "WB");
    if (muA) check_shape(*muA, {in.CJ}, "muA");
    if (muB) check_shape(*muB, {in.CJ}, "muB");
    if (pointsI.has_value() != pointsJ.has_value())
      throw std::invalid_argument("pointsI and pointsJ must be given together");
    if (pointsI) {
      if (pointsI->ndim() != 2) throw std::invalid_argument("pointsI must have shape (N, D)");
      in.NP = pointsI->shape(0);
      check_shape(*pointsI, {in.NP, D}, "pointsI");
      check_shape(*pointsJ, {in.NP, D}, "pointsJ");
      in.pointsI = pointsI->data();
      in.pointsJ = pointsJ->data();
    }
    in.I = I_.data();
    in.J = J_.data();
    in.v = v.data();
    in.L = L.data();
    in.T = T.data();
    in.K = K.data();
    in.LL = LL.data();
    in.WM = WM.data();
    in.WA = WA.data();
    in.WB = WB.data();
    in.muA = muA ? muA->data() : nullptr;
    in.muB = muB ? muB->data() : nullptr;
    in.params.sigmaM = sigmaM;
    in.params.sigmaA = sigmaA;
    in.params.sigmaB = sigmaB;
    in.params.sigmaR = sigmaR;
    in.params.sigmaP = sigmaP;
    in.params.epL = epL;
    in.params.epT = epT;
    in.params.epV = epV;
    in.params.diffeo_start = diffeo_start;
    in.params.DV = DV;
    if (affine_step_control &&
        !(step_shrink > 0.0 && step_shrink < 1.0 && step_grow >= 1.0 && step_floor > 0.0 &&
          step_floor <= step_cap && step_cap <= 1.0))
      throw std::invalid_argument(
          "affine step control needs 0 < step_shrink < 1, step_grow >= 1 and "
          "0 < step_floor <= step_cap <= 1");
    in.params.affine_step_control = affine_step_control;
    in.params.step_shrink = step_shrink;
    in.params.step_grow = step_grow;
    in.params.step_floor = step_floor;
    in.params.step_cap = step_cap;
    in.params.landmark_exact = landmark_exact;
    if (mixture) {
      if (mix_classes != "M" && mix_classes != "MB" && mix_classes != "MA" && mix_classes != "MAB")
        throw std::invalid_argument("mix_classes must be 'M', 'MB', 'MA' or 'MAB'");
      if (!muA || !muB) throw std::invalid_argument("the mixture EM needs the initial means muA and muB");
    }
    in.params.mixture = mixture;
    in.params.mix_start = mix_start;
    in.params.mix_every = mix_every;
    in.params.mix_classA = mix_classes.find('A') != std::string::npos;
    in.params.mix_classB = mix_classes.find('B') != std::string::npos;
    in.params.mix_est_muA = mix_est_muA;
    in.params.mix_est_muB = mix_est_muB;
    in.params.mix_pi_min = mix_pi_min;
    for (int k = 0; k < 3; ++k) in.params.mix_pi_init[k] = mix_pi_init[k];
    e_ = std::make_unique<stalign::Engine<D>>(in, num_threads);
  }

  void run(int n) { e_->run(n); }

  py::dict gradient() {
    std::vector<double> vg, dir;
    {
      py::gil_scoped_release release;
      e_->gradient(vg, dir);
    }
    std::vector<double> gl(D * D), gt(D);
    for (int a = 0; a < D; ++a) {
      for (int b = 0; b < D; ++b) gl[a * D + b] = e_->gL[a][b];
      gt[a] = e_->gT[a];
    }
    return py::dict("Lgrad"_a = to_numpy(std::move(gl), {D, D}),
                    "Tgrad"_a = to_numpy(std::move(gt), {D}),
                    "vgrad"_a = to_numpy(std::move(vg), vshape()),
                    "direction"_a = to_numpy(std::move(dir), vshape()), "EM"_a = e_->E_last[1],
                    "ER"_a = e_->E_last[2], "EP"_a = e_->E_last[3]);
  }

  int iteration() const { return e_->it; }
  long long landmark_unconverged() const { return e_->lm_unconverged; }
  py::array_t<double> L() const {
    std::vector<double> o(D * D);
    for (int a = 0; a < D; ++a)
      for (int b = 0; b < D; ++b) o[a * D + b] = e_->L[a][b];
    return to_numpy(std::move(o), {D, D});
  }
  py::array_t<double> T() const {
    return to_numpy(std::vector<double>(e_->T, e_->T + D), {D});
  }
  py::array_t<double> A_eval() const {
    const double* a = &e_->A_eval[0][0];
    return to_numpy(std::vector<double>(a, a + (D + 1) * (D + 1)), {D + 1, D + 1});
  }
  py::array_t<double> v() const { return to_numpy(std::vector<double>(e_->v), vshape()); }
  py::array_t<double> v_eval() const {
    return to_numpy(std::vector<double>(e_->v_eval), vshape());
  }
  py::array_t<double> WM() const { return to_numpy(std::vector<double>(e_->WM), jshape()); }
  py::array_t<double> WA() const { return to_numpy(std::vector<double>(e_->WA), jshape()); }
  py::array_t<double> WB() const { return to_numpy(std::vector<double>(e_->WB), jshape()); }
  // deformed source (C_I, *nJ) of the last forward pass
  py::array_t<double> AI() const {
    const auto& E = *e_;
    std::vector<double> o(E.CI * E.NJ);
    for (i64 j = 0; j < E.NJ; ++j)
      for (int c = 0; c < E.CI; ++c) o[c * E.NJ + j] = E.AIbuf[j * E.CI + c];
    return to_numpy(std::move(o), jshape(E.CI));
  }
  // its contrast fit (C_J, *nJ)
  py::array_t<double> fAI() const {
    const auto& E = *e_;
    std::vector<double> o(E.CJ * E.NJ);
    for (i64 j = 0; j < E.NJ; ++j)
      for (int c = 0; c < E.CJ; ++c) {
        double f = E.coeffs[c];
        for (int i = 0; i < E.CI; ++i) f += E.AIbuf[j * E.CI + i] * E.coeffs[(1 + i) * E.CJ + c];
        o[c * E.NJ + j] = f;
      }
    return to_numpy(std::move(o), jshape(E.CJ));
  }
  // sample points (*nJ, D) of the last forward pass
  py::array_t<double> Xs() const {
    const auto& E = *e_;
    std::vector<double> o(E.NJ * D);
    for (i64 j = 0; j < E.NJ; ++j)
      for (int d = 0; d < D; ++d) o[j * D + d] = E.Xbuf[j * (E.nt + 1) * D + d];
    shape_t s = jshape();
    s.push_back(D);
    return to_numpy(std::move(o), s);
  }
  // transformed pointsI (N, D) of the last forward pass
  py::array_t<double> pointsIt() const {
    return to_numpy(std::vector<double>(e_->pIt), {(py::ssize_t)e_->pIt.size() / D, D});
  }
  // affine step multipliers (D*D + D: L row-major, then T); all 1 without step control
  py::array_t<double> step_multipliers() const {
    const double* m = e_->step_m;
    return to_numpy(std::vector<double>(m, m + D * D + D), {D * D + D});
  }
  // [E, EM, ER, EP] of every iteration (E = EM + ER, as in the original's Esave)
  py::array_t<double> Esave() const {
    return to_numpy(std::vector<double>(e_->Ehist), {(py::ssize_t)e_->Ehist.size() / 4, 4});
  }

 private:
  shape_t vshape() const {
    shape_t s{e_->nt};
    for (int d = 0; d < D; ++d) s.push_back(e_->gV.n[d]);
    s.push_back(D);
    return s;
  }
  shape_t jshape(py::ssize_t lead = 0) const {
    shape_t s;
    if (lead) s.push_back(lead);
    for (int d = 0; d < D; ++d) s.push_back(e_->m[d]);
    return s;
  }

  carr I_, J_;  // the engine reads these in place: keep them alive
  std::unique_ptr<stalign::Engine<D>> e_;
};

template <int D>
void bind_engine(py::module_& m, const char* name, const char* doc) {
  using E = PyEngine<D>;
  // module_local: the module may be loaded twice in one process (as STalign._stalign_cpp and,
  // with the tutorials' sys.path import, as _stalign_cpp); each copy registers its own types.
  py::class_<E>(m, name, doc, py::module_local())
      .def(py::init<carr, const std::vector<carr>&, carr, const std::vector<carr>&, carr,
                    const std::vector<carr>&, carr, carr, carr, carr, carr, carr, carr,
                    std::optional<carr>, std::optional<carr>, std::optional<carr>,
                    std::optional<carr>, double, double, double, double, double, double, double,
                    double, int, double, bool, double, double, double, double, int, bool, bool, int,
                    int, const std::string&, double, std::array<double, 3>, bool, bool>(),
           "I"_a, "xI"_a, "J"_a, "xJ"_a, "v"_a, "xv"_a, "L"_a, "T"_a, "K"_a, "LL"_a, "WM"_a,
           "WA"_a, "WB"_a, py::kw_only(), "muA"_a = py::none(), "muB"_a = py::none(),
           "pointsI"_a = py::none(), "pointsJ"_a = py::none(), "sigmaM"_a, "sigmaA"_a,
           "sigmaB"_a, "sigmaR"_a, "sigmaP"_a, "epL"_a, "epT"_a, "epV"_a, "diffeo_start"_a,
           "DV"_a, "affine_step_control"_a = false, "step_shrink"_a = 0.5, "step_grow"_a = 1.05,
           "step_floor"_a = 1e-3, "step_cap"_a = 1.0, "num_threads"_a = 1, "landmark_exact"_a = true,
           "mixture"_a = false, "mix_start"_a = 50, "mix_every"_a = 5, "mix_classes"_a = "MB",
           "mix_pi_min"_a = 0.5, "mix_pi_init"_a = std::array<double, 3>{0.9, 0.05, 0.05},
           "mix_est_muA"_a = true, "mix_est_muB"_a = false)
      .def("run", &E::run, "n"_a, py::call_guard<py::gil_scoped_release>(),
           "Advance n iterations (the iteration counter persists across calls).")
      .def("gradient", &E::gradient,
           "Gradients at the current state without updating it (dict: Lgrad, Tgrad, "
           "vgrad = d(EM+EP)/dv, direction = K*dE/dv including ER, EM, ER, EP).")
      .def_property_readonly("iteration", &E::iteration)
      .def_property_readonly("landmark_unconverged", &E::landmark_unconverged,
                             "Exact landmark solves (landmark, time step) that did not converge.")
      .def_property_readonly("L", &E::L)
      .def_property_readonly("T", &E::T)
      .def_property_readonly("A_eval", &E::A_eval)
      .def_property_readonly("v", &E::v)
      .def_property_readonly("v_eval", &E::v_eval)
      .def_property_readonly("WM", &E::WM)
      .def_property_readonly("WA", &E::WA)
      .def_property_readonly("WB", &E::WB)
      .def_property_readonly("AI", &E::AI)
      .def_property_readonly("fAI", &E::fAI)
      .def_property_readonly("Xs", &E::Xs)
      .def_property_readonly("pointsIt", &E::pointsIt)
      .def_property_readonly("step_multipliers", &E::step_multipliers)
      .def_property_readonly("Esave", &E::Esave);
}

}  // namespace

PYBIND11_MODULE(_stalign_cpp, m) {
  m.doc() = "C++ CPU (float64) engine for STalign's LDDMM and LDDMM_3D_to_slice";
  // bumped whenever the binding interface changes, so a stale in-place build is detected
  m.attr("API_VERSION") = 4;
  m.attr("MAX_CI") = stalign::MAX_CI;
  m.attr("MAX_CJ") = stalign::MAX_CJ;
  m.attr("MAX_NT") = stalign::MAX_NT;
  bind_engine<2>(m, "Engine2D", "LDDMM engine for 2D source and target images.");
  bind_engine<3>(m, "Engine3D", "LDDMM engine for a 3D source volume and a 2D target slice.");
}
