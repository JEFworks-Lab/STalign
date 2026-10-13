"""Tests of the STalign helpers around rasterization: rasterizePCA, make_scree,
saveRasters, calculate_tre, the Allen atlas downloads, the brain region plots
and the point / curve annotator scripts.
"""
import builtins
import glob
import http.server
import os
import runpy
import sys
import threading

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from STalign import STalign  # noqa: E402

pd = pytest.importorskip('pandas')
torch = pytest.importorskip('torch')
requests = pytest.importorskip('requests')


def cells(n, seed=0, extent=3000.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, extent, n), rng.uniform(0, extent, n)


# ---------------------------------------------------------------------------
# rasterizePCA, make_scree, saveRasters
# ---------------------------------------------------------------------------
def gene_table(n, seed=0):
    rng = np.random.default_rng(seed)
    G = pd.DataFrame({f'gene{i}': rng.poisson(1.0 + i, n) for i in range(4)})
    G['absent'] = 0          # an all-zero gene (common in crops / control probes)
    G['constant'] = 2.0      # a constant gene
    return G


def test_rasterizePCA():
    x, y = cells(600, seed=1)
    G = gene_table(x.size, seed=1)
    X, Y, W, V, Z, nrows, ncols = STalign.rasterizePCA(x, y, G)
    assert X.shape == (6, nrows, ncols) and Y.shape == (6, nrows*ncols)
    assert np.isfinite(X).all() and np.isfinite(W).all() and np.isfinite(V).all()
    for i in range(G.shape[1]):
        XI, YI, I = STalign.rasterize(x, y, np.array(G.iloc[:, i]), dx=30.0, blur=1.0, draw=0)
        np.testing.assert_array_equal(X[i], I[0])
    assert (X[4] == 0).all() and (X[5] == 0).all()  # constant genes give zero images
    np.testing.assert_allclose(Y.mean(1), 0, atol=1e-12)
    assert (np.diff(W) <= 1e-12).all()  # descending
    np.testing.assert_allclose(V.T @ V, np.eye(6), atol=1e-12)
    np.testing.assert_allclose(Z, V.T @ Y)


def test_make_scree():
    fig = STalign.make_scree(np.array([3.0, 2.0, 1.0, 0.5, 0.2, 0.1]), 'test')
    assert isinstance(fig, matplotlib.figure.Figure)
    with pytest.raises(ValueError):
        STalign.make_scree(np.array([3.0, 2.0]), 'test', p=6)
    plt.close('all')


def test_saveRasters_folder_without_trailing_slash(tmp_path):
    x, y = cells(400, seed=2)
    X, Y, W, V, Z, nrows, ncols = STalign.rasterizePCA(x, y, gene_table(x.size, seed=2))
    scree = STalign.make_scree(W, 'demo')
    out = tmp_path/'rasters'  # does not exist yet, no trailing slash
    STalign.saveRasters(X, Y, W, V, Z, scree, nrows, ncols, 'demo', str(out))
    assert sorted(os.listdir(out)) == ['demo_arrays.npz', 'demo_rgbfig.jpg', 'demo_screefig.jpg']
    assert not list(tmp_path.glob('rastersdemo*'))
    with np.load(out/'demo_arrays.npz') as f:
        np.testing.assert_array_equal(f['W'], W)
        assert f['I_pca'].shape == (6, nrows, ncols)
    plt.close('all')


# ---------------------------------------------------------------------------
# calculate_tre
# ---------------------------------------------------------------------------
def test_calculate_tre_inputs():
    a = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 5.0]])
    b = np.array([[3.0, 4.0], [1.0, 1.0], [2.0, 6.0]])
    d = np.array([5.0, 0.0, 1.0])
    expected = (np.mean(d), np.std(d))
    for p, q in [(a, b), (torch.tensor(a), b), (a, torch.tensor(b)), (torch.tensor(a), torch.tensor(b)),
                 (a.astype(int), b.astype(int)), (torch.tensor(a, dtype=torch.float32, requires_grad=True), b),
                 (a.tolist(), b.tolist())]:
        np.testing.assert_allclose(STalign.calculate_tre(p, q), expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Allen atlas downloads (local HTTP server, no network)
# ---------------------------------------------------------------------------
@pytest.fixture
def server(monkeypatch):
    """A local HTTP server; set routes[path] = (status, body)."""
    for k in ('NO_PROXY', 'no_proxy'):
        monkeypatch.setenv(k, '127.0.0.1,localhost')
    routes = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            status, body = routes.get(self.path, (404, b'not found'))
            self.send_response(status)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{srv.server_address[1]}', routes
    srv.shutdown()
    srv.server_close()


def test_download_aba_ontology(server, tmp_path):
    url, routes = server
    routes['/ontology.csv'] = (200, b'id,acronym,name\n315,Isocortex,Isocortex\n382,CA1,Field CA1\n')
    routes['/down.csv'] = (503, b'<html>Service Unavailable</html>')
    name, namesdict = STalign.download_aba_ontology(url + '/ontology.csv', str(tmp_path/'ontology.csv'))
    assert name == str(tmp_path/'ontology.csv')
    assert namesdict[382] == 'CA1' and namesdict[0] == 'bg' and namesdict[1] == 'unk'
    with pytest.raises(requests.HTTPError):
        STalign.download_aba_ontology(url + '/down.csv', str(tmp_path/'down.csv'))
    assert not (tmp_path/'down.csv').exists()


def test_download_aba_image_labels(server, tmp_path):
    url, routes = server
    image, labels = os.urandom(3000), os.urandom(5000)
    routes['/image.nrrd'] = (200, image)
    routes['/labels.nrrd'] = (200, labels)
    routes['/down.nrrd'] = (503, b'<html>Service Unavailable</html>')
    imagefile, labelfile = str(tmp_path/'image.nrrd'), str(tmp_path/'labels.nrrd')
    out = STalign.download_aba_image_labels(url + '/image.nrrd', url + '/labels.nrrd', imagefile, labelfile)
    assert out == (imagefile, labelfile)
    assert open(imagefile, 'rb').read() == image and open(labelfile, 'rb').read() == labels
    with pytest.raises(requests.HTTPError):
        STalign.download_aba_image_labels(url + '/image.nrrd', url + '/down.nrrd', imagefile,
                                          str(tmp_path/'down.nrrd'))
    assert sorted(os.listdir(tmp_path)) == ['image.nrrd', 'labels.nrrd']


# ---------------------------------------------------------------------------
# brain region plots
# ---------------------------------------------------------------------------
def region_table(nregions, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'x': rng.uniform(0, 1, n), 'y': rng.uniform(0, 1, n),
                         'acronym': [f'R{i}' for i in rng.integers(0, nregions, n)]})


@pytest.mark.parametrize('nregions', [3, 15, 60])
def test_plot_brain_regions_distinct_colors(nregions):
    fig, ax = STalign.plot_brain_regions(region_table(nregions))
    colors = {tuple(np.round(c.get_facecolor()[0], 6)) for c in ax.collections}
    assert len(ax.collections) == nregions and len(colors) == nregions
    assert len(ax.get_legend().get_texts()) == nregions
    plt.close('all')


def test_plot_subset_brain_regions():
    fig, ax = STalign.plot_subset_brain_regions(region_table(10), ['R1', 'R2'])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['R1', 'R2']
    plt.close('all')


# ---------------------------------------------------------------------------
# annotator scripts (run as __main__ with scripted input and clicks)
# ---------------------------------------------------------------------------
def run_annotator(monkeypatch, script, argv, names, clicks, glob_result=None):
    clicks = list(clicks)
    names = list(names)
    monkeypatch.setattr(builtins, 'input', lambda prompt='': names.pop(0) if names else '')
    monkeypatch.setattr(plt, 'ginput', lambda *a, **k: clicks.pop(0) if clicks else [])
    monkeypatch.setattr(plt, 'show', lambda *a, **k: None)
    monkeypatch.setattr(plt, 'pause', lambda *a, **k: None)
    if glob_result is not None:
        monkeypatch.setattr(glob, 'glob', lambda pattern: list(glob_result))
    monkeypatch.setattr(sys, 'argv', [script] + argv)
    try:
        runpy.run_path(os.path.join(REPO, 'STalign', script), run_name='__main__')
    finally:
        plt.close('all')


@pytest.fixture
def images(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ('src', 'tgt'):
        np.savez(name, x=np.arange(10.0)*50, y=np.arange(8.0)*50, I=np.random.rand(3, 8, 10))
    return tmp_path


def test_point_annotator_output_names(monkeypatch, images):
    # 5 clicks: the last, unpaired one is dropped
    clicks = [[(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]]
    run_annotator(monkeypatch, 'point_annotator.py', ['src.npz', 'tgt.npz', 'oS.npy', 'oT.npy'], ['a'], clicks)
    S = np.load('oS.npy', allow_pickle=True).item()
    T = np.load('oT.npy', allow_pickle=True).item()
    assert S == {'a': [(10, 10), (30, 30)]} and T == {'a': [(20, 20), (40, 40)]}
    # one output name: the target name is the default
    run_annotator(monkeypatch, 'point_annotator.py', ['src.npz', 'tgt.npz', 'oS2.npy'], ['b'], [[(1, 2), (3, 4)]])
    assert os.path.exists('oS2.npy') and os.path.exists('tgt_points.npy')


def test_point_annotator_defaults_and_numeric_source(monkeypatch, images):
    run_annotator(monkeypatch, 'point_annotator.py', ['src.npz', 'tgt.npz'], ['a'], [[(1, 2), (3, 4)]])
    assert np.load('src_points.npy', allow_pickle=True).item() == {'a': [(1, 2)]}
    assert np.load('tgt_points.npy', allow_pickle=True).item() == {'a': [(3, 4)]}
    # a numeric source picks a file from a list (it used to fail to load it)
    os.remove('src_points.npy')
    run_annotator(monkeypatch, 'point_annotator.py', ['0', 'tgt.npz'], ['a'], [[(5, 6), (7, 8)]],
                  glob_result=[str(images/'src.npz')])
    assert np.load('src_points.npy', allow_pickle=True).item() == {'a': [(5, 6)]}


def test_curve_annotator_output_name(monkeypatch, images):
    run_annotator(monkeypatch, 'curve_annotator.py', ['src.npz', '-o', 'curves.npy'], ['c'], [[(1, 1), (2, 3), (4, 4)]])
    assert np.load('curves.npy', allow_pickle=True).item() == {'c': [(1, 1), (2, 3), (4, 4)]}
    run_annotator(monkeypatch, 'curve_annotator.py', ['src.npz'], ['d'], [[(1, 1), (2, 2)]])
    assert np.load('src_curves.npy', allow_pickle=True).item() == {'d': [(1, 1), (2, 2)]}
