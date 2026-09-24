from pathlib import Path
from utils.pipeline import find_project_root
from utils.reporting import build_progress_markdown


def test_find_project_root_from_various_starting_points():
    here = Path(__file__).resolve()
    root_from_file = find_project_root(here)
    root_from_dir = find_project_root(here.parent)
    assert root_from_file == root_from_dir
    assert (root_from_file / "1_scripts").is_dir()
    assert (root_from_file / "2_pipeline").is_dir()


def test_build_progress_markdown_missing_step_raises(tmp_path):
    (tmp_path / "1_scripts").mkdir()
    (tmp_path / "2_pipeline").mkdir()
    try:
        build_progress_markdown("no_such_step", project_root=tmp_path)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as e:
        assert "no_such_step" in str(e)


def test_build_progress_markdown_accepts_str_or_path(tmp_path):
    """Regression test: project_root must work whether passed as a str
    or a pathlib.Path -- this is exactly the bug that motivated the
    defensive Path(...) coercion in build_progress_markdown."""
    (tmp_path / "1_scripts").mkdir()
    step_out = tmp_path / "2_pipeline" / "01_fake_step" / "out"
    step_out.mkdir(parents=True)
    (step_out / "fig1.png").write_bytes(b"placeholder")

    md_path_from_path = build_progress_markdown("01_fake_step", project_root=tmp_path)
    assert md_path_from_path.exists()

    md_path_from_str = build_progress_markdown("01_fake_step", project_root=str(tmp_path))
    assert md_path_from_str.exists()
    assert md_path_from_path == md_path_from_str


def test_build_progress_markdown_embeds_figure(tmp_path):
    (tmp_path / "1_scripts").mkdir()
    step_out = tmp_path / "2_pipeline" / "01_fake_step" / "out"
    step_out.mkdir(parents=True)
    (step_out / "fig1.png").write_bytes(b"placeholder")

    md_path = build_progress_markdown("01_fake_step", project_root=tmp_path)
    content = md_path.read_text()
    assert "Fig1" in content or "fig1" in content
    assert "01_fake_step/out/fig1.png" in content


def test_build_progress_markdown_embeds_csv_table(tmp_path):
    (tmp_path / "1_scripts").mkdir()
    step_out = tmp_path / "2_pipeline" / "01_fake_step" / "out"
    step_out.mkdir(parents=True)
    (step_out / "summary.csv").write_text("seed,accuracy\n0,0.9\n1,0.85\n")

    md_path = build_progress_markdown("01_fake_step", project_root=tmp_path)
    content = md_path.read_text()
    assert "| seed | accuracy |" in content
    assert "0.9" in content


def test_build_progress_markdown_empty_out_says_so(tmp_path):
    (tmp_path / "1_scripts").mkdir()
    (tmp_path / "2_pipeline" / "01_fake_step" / "out").mkdir(parents=True)
    md_path = build_progress_markdown("01_fake_step", project_root=tmp_path)
    assert "nothing to report yet" in md_path.read_text()
