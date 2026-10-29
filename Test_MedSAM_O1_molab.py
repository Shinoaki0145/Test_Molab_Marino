"""SemiSAM-O1 (Molab / marimo): adaptation of Test_MedSAM_O1.ipynb.

Open this as a marimo notebook on molab, not as a plain Python script.
The setup and training operations are gated behind Run buttons.

Important:
- In Molab's developer panel > Secrets, add GITHUB_TOKEN and HF_TOKEN to .env.
- Install python-dotenv through the package manager if it isn't available.
- Enable GPU using the notebook specs menu.
- Files downloaded to the Molab runtime are not guaranteed to persist.
- Connect Google Drive and click Sync after training to copy results.
"""

import marimo

app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import os
    import subprocess
    from pathlib import Path
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import time
    import shutil

    return mo, os, subprocess, Path, datetime, ZoneInfo, time, shutil


@app.cell
def _(mo):
    mo.md(
        """
        # SemiSAM-O1 · Molab

        Chuyển từ `Test_MedSAM_O1.ipynb` chạy trên Google Colab.
        Lần lượt bấm các nút **Clone → Conda → Download → Prepare → Train → Sync**.

        **Trước khi bắt đầu:** bật GPU; trong **Developer panel → Secrets**
        thêm `GITHUB_TOKEN` và `HF_TOKEN` (không điền token vào code).
        Cần `python-dotenv` để đọc `.env`.

        Cảnh báo: Molab có giới hạn phiên chạy và hạn chế tác vụ không tương tác;
        việc chạy train lâu có thể bị dừng. File checkpoint chỉ được đảm bảo
        sau khi sao lưu ra Drive. Không bật autorun cho các bước tốn tài nguyên.
        """
    )
    return


@app.cell
def _(Path, mo, os):
    BASE = Path(mo.notebook_dir()).resolve() / "semisam_work"
    REPO = BASE / "SemiSam-O1"
    MINICONDA = BASE / "miniconda3"
    CONDA_ENV = BASE / "semisam-o1"
    DOWNLOADS = BASE / "downloads"
    BASE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(BASE / ".cache" / "huggingface"))
    return BASE, REPO, MINICONDA, CONDA_ENV, DOWNLOADS


@app.cell
def _(Path, mo):
    def load_secrets():
        try:
            from dotenv import load_dotenv
        except ImportError as error:
            raise RuntimeError(
                "Vào Package Manager của Molab, cài python-dotenv trước."
            ) from error
        # Molab developer panel keeps .env protected when a notebook is forked.
        load_dotenv(dotenv_path=Path(mo.notebook_dir()) / ".env", override=False)
    return (load_secrets,)


@app.cell
def _(subprocess, os):
    _gpu = subprocess.run(
        ["nvidia-smi"], capture_output=True, text=True, check=False
    )
    print(_gpu.stdout if _gpu.returncode == 0 else (
        "Chưa thấy nvidia-smi. Bật GPU bằng notebook specs rồi chạy lại.\n"
        + _gpu.stderr
    ))
    return


@app.cell
def _(mo):
    clone_btn = mo.ui.run_button(label="1. Clone branch addDINOv3")
    clone_btn
    return (clone_btn,)


@app.cell
def _(clone_btn, mo, REPO, BASE, os, subprocess, load_secrets):
    mo.stop(not clone_btn.value)
    load_secrets()
    _token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not _token:
        raise RuntimeError("Thiếu GITHUB_TOKEN trong Secrets/.env của Molab.")
    if REPO.exists():
        _branch = subprocess.run(
            ["git", "-C", str(REPO), "branch", "--show-current"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if _branch != "addDINOv3":
            raise RuntimeError(f"Repo đã tồn tại nhưng đang ở branch {_branch!r}.")
        print("Đã có repository đúng branch:", REPO)
    else:
        _askpass = BASE / "git-askpass.sh"
        _askpass.write_text(
            '#!/bin/sh\n'
            'case "$1" in\n'
            '  *Username*) echo x-access-token ;;\n'
            '  *) printf "%s\\n" "$GITHUB_TOKEN" ;;\n'
            'esac\n'
        )
        _askpass.chmod(0o700)
        try:
            _env = {**os.environ, "GIT_ASKPASS": str(_askpass),
                    "GIT_TERMINAL_PROMPT": "0"}
            subprocess.run(
                ["git", "clone", "--branch", "addDINOv3", "--single-branch",
                 "https://github.com/ThienNhanHoangHuyKLTN/SemiSam-O1.git",
                 str(REPO)],
                env=_env, check=True,
            )
        finally:
            _askpass.unlink(missing_ok=True)
        print("Clone thành công:", REPO)
    return


@app.cell
def _(mo):
    conda_btn = mo.ui.run_button(label="2. Cài môi trường từ conda.yaml")
    conda_btn
    return (conda_btn,)


@app.cell
def _(conda_btn, mo, MINICONDA, CONDA_ENV, REPO, BASE, Path, subprocess):
    mo.stop(not conda_btn.value)
    if not (REPO / "conda.yaml").is_file():
        raise FileNotFoundError("Chưa có conda.yaml. Bấm Clone trước.")
    _conda = MINICONDA / "bin" / "conda"
    if not _conda.exists():
        import urllib.request
        _installer = BASE / "miniconda.sh"
        urllib.request.urlretrieve(
            "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh",
            str(_installer),
        )
        subprocess.run(
            ["bash", str(_installer), "-b", "-p", str(MINICONDA)],
            check=True,
        )
        _installer.unlink(missing_ok=True)
    for _channel in ("main", "r"):
        subprocess.run(
            [str(_conda), "tos", "accept", "--override-channels",
             "--channel", f"https://repo.anaconda.com/pkgs/{_channel}"],
            check=True,
        )
    if not (CONDA_ENV / "bin" / "python").exists():
        subprocess.run(
            [str(_conda), "env", "create", "-p", str(CONDA_ENV),
             "-f", "conda.yaml"],
            cwd=REPO, check=True,
        )
    _python = CONDA_ENV / "bin" / "python"
    subprocess.run(
        [str(_python), "-m", "pip", "install", "-U", "huggingface_hub"],
        check=True,
    )
    subprocess.run([str(_python), "-m", "pip", "check"], check=True)
    print("Môi trường:", CONDA_ENV)
    print("Lưu ý: môi trường Torch/CUDA trong conda.yaml có thể không hỗ trợ GPU Blackwell.")
    return


@app.cell
def _(mo):
    download_btn = mo.ui.run_button(label="3. Tải LA, DINOv3, SAM-Med3D")
    download_btn
    return (download_btn,)


@app.cell
def _(download_btn, mo, CONDA_ENV, REPO, DOWNLOADS, os, subprocess, load_secrets):
    mo.stop(not download_btn.value)
    load_secrets()
    _hf = CONDA_ENV / "bin" / "hf"
    if not _hf.is_file():
        raise FileNotFoundError("Không có hf CLI trong conda env. Bấm Conda trước.")
    _env = dict(os.environ)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    (REPO / "data").mkdir(exist_ok=True)
    _la = REPO / "data" / "LA" / "train.txt"
    if not _la.is_file():
        subprocess.run(
            [str(_hf), "download", "SemiSamO1/DATASET", "LA.zip",
             "--repo-type", "dataset", "--local-dir", str(DOWNLOADS)],
            env=_env, check=True,
        )
        subprocess.run(
            ["unzip", "-q", "-o", str(DOWNLOADS / "LA.zip"),
             "-d", str(REPO / "data")], check=True,
        )
    _ckpt = REPO / "code" / "pretrained_ckpt"
    _ckpt.mkdir(parents=True, exist_ok=True)
    _models = (
        ("SemiSamO1/Generalist", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
        ("blueyo0/SAM-Med3D", "sam_med3d_turbo.pth"),
    )
    for _repo_id, _name in _models:
        if not (_ckpt / _name).is_file() or (_ckpt / _name).stat().st_size == 0:
            subprocess.run(
                [str(_hf), "download", _repo_id, _name,
                 "--local-dir", str(_ckpt)],
                env=_env, check=True,
            )
    for _file in (
        REPO / "data" / "LA" / "train.txt",
        REPO / "data" / "LA" / "val.txt",
        _ckpt / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        _ckpt / "sam_med3d_turbo.pth",
    ):
        if not _file.is_file() or _file.stat().st_size == 0:
            raise FileNotFoundError(f"Thiếu dữ liệu/model: {_file}")
    print("Tải dữ liệu và trọng số xong.")
    return


@app.cell
def _(mo):
    prepare_btn = mo.ui.run_button(label="4. Tạo thư mục kết quả")
    prepare_btn
    return (prepare_btn,)


@app.cell
def _(prepare_btn, mo, BASE, REPO, datetime, ZoneInfo):
    mo.stop(not prepare_btn.value)
    RUN_NAME = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = BASE / "runs" / RUN_NAME
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _model_link = REPO / "model"
    if _model_link.exists() or _model_link.is_symlink():
        if _model_link.is_symlink() and _model_link.resolve() == OUTPUT_DIR:
            print("Symlink đã tồn tại đúng vị trí.")
        else:
            raise RuntimeError(
                f"{_model_link} đã tồn tại. Không tự ghi đè. "
                "Xử lý thư mục/symlink hiện tại trước khi tạo run mới."
            )
    else:
        _model_link.symlink_to(OUTPUT_DIR, target_is_directory=True)
    print("Run:", RUN_NAME)
    print("Checkpoint cục bộ:", OUTPUT_DIR)
    return RUN_NAME, OUTPUT_DIR


@app.cell
def _(mo):
    train_btn = mo.ui.run_button(label="5. Train SemiSAM-O1 (15,000 iterations)", kind="danger")
    train_btn
    return (train_btn,)


@app.cell
def _(train_btn, mo, REPO, CONDA_ENV, OUTPUT_DIR, subprocess, os):
    mo.stop(not train_btn.value)
    _train_script = REPO / "code" / "train_SemiSAM_O1.py"
    if not _train_script.is_file():
        raise FileNotFoundError("Chưa clone repo hoặc thiếu train script.")
    _python = CONDA_ENV / "bin" / "python"
    _check = (
        "import torch, torchvision; "
        "from pathlib import Path; "
        "assert torch.cuda.is_available(), 'GPU CUDA unavailable'; "
        "assert (torch.ones(2, device='cuda') * 2).tolist() == [2.0, 2.0]; "
        "torch.cuda.synchronize(); "
        "assert Path('third_party/dinov3/hubconf.py').is_file(); "
        "print('GPU:', torch.cuda.get_device_name(0)); "
        "print('VRAM (GiB):', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)); "
        "print('torch:', torch.__version__, 'torchvision:', torchvision.__version__)"
    )
    subprocess.run(
        [str(_python), "-c", _check],
        cwd=REPO / "code", env=dict(os.environ), check=True,
    )
    _cmd = [
        str(_python), "-u", str(_train_script),
        "--root_path", "../data/LA",
        "--exp", "SemiSAM_O1/LA_DINO_SAM_MT",
        "--backbone", "mt",
        "--round0_method", "dino_sam",
        "--round0_roi_mode", "seeded",
        "--round0_threshold_mode", "reference_fallback",
        "--pseudo_max_weight", "1.0",
        "--pseudo_ignore_empty",
        "--num_workers", "2",
        "--max_iterations", "15000",
        "--num_rounds", "3",
        "--seed", "1337",
    ]
    _log = OUTPUT_DIR / "train.log"
    print("Đang huấn luyện. Log:", _log, flush=True)
    with _log.open("a", encoding="utf-8", buffering=1) as _fh:
        _process = subprocess.Popen(
            _cmd, cwd=REPO / "code", env=dict(os.environ),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        try:
            for _line in _process.stdout:
                _fh.write(_line)
                print(_line, end="", flush=True)
            _status = _process.wait()
        except BaseException:
            _process.terminate()
            raise
    if _status:
        raise RuntimeError(f"Train failed: return code {_status}. Xem {_log}")
    print("Đã train xong. Nhớ bấm Sync để sao lưu checkpoint lên Drive.")
    return


@app.cell
def _(mo):
    drive_btn = mo.ui.run_button(label="6a. Kết nối Google Drive")
    drive_btn
    return (drive_btn,)


@app.cell
def _(drive_btn, mo):
    mo.stop(not drive_btn.value)
    try:
        from gdrive_fsspec import GoogleDriveFileSystem
    except ImportError as error:
        raise RuntimeError("Cài gdrive-fsspec bằng Package Manager của Molab.") from error
    drive_fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True)
    print("Google Drive đã kết nối. Kiểm tra thư mục root trong Remote Storage panel.")
    return (drive_fs,)


@app.cell
def _(mo):
    sync_btn = mo.ui.run_button(label="6b. Sao lưu checkpoint lên Google Drive")
    sync_btn
    return (sync_btn,)


@app.cell
def _(sync_btn, mo, drive_fs, OUTPUT_DIR, RUN_NAME):
    mo.stop(not sync_btn.value)
    _remote_base = f"SemiSam-O1/runs/{RUN_NAME}"
    drive_fs.makedirs(_remote_base, exist_ok=True)
    _files = sorted(_p for _p in OUTPUT_DIR.rglob("*") if _p.is_file())
    if not _files:
        raise RuntimeError("Không tìm thấy file output để sao lưu.")
    import posixpath
    for _local in _files:
        _remote = f"{_remote_base}/{_local.relative_to(OUTPUT_DIR).as_posix()}"
        _remote_parent = posixpath.dirname(_remote)
        drive_fs.makedirs(_remote_parent, exist_ok=True)
        drive_fs.put_file(str(_local), _remote)
        print(f"Đã tải: {_local.relative_to(OUTPUT_DIR)}")
    print("Google Drive:", _remote_base)
    return


if __name__ == "__main__":
    app.run()
