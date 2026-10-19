#!/usr/bin/env bash
# Download or prepare the datasets under $POINTMIL_DATA (default ./data).
# Usage: bash scripts/download_data.sh [intra|rbc|modelnet40|scanobjectnn|atlas1|all]
set -euo pipefail
ROOT=${POINTMIL_DATA:-$(pwd)/data}
mkdir -p "$ROOT"
what=${1:-all}

intra() {
  echo "== IntrA (Yang et al., CVPR 2020)"
  echo "The dataset is distributed by its authors through a Google Drive folder:"
  echo "  https://drive.google.com/drive/folders/1yjLdofRRqyklgwFOC0K4r7ee1LPKstPh"
  echo "  (project page: https://github.com/intra3d2019/IntrA; please cite their paper)"
  echo "Download the folders 'annotated' and 'generated' and place them so that you have:"
  echo "  $ROOT/IntrA/annotated/ad/*.ad   $ROOT/IntrA/generated/vessel/ad/*.ad   $ROOT/IntrA/generated/aneurysm/ad/*.ad"
  echo "With gdown installed (pip install gdown) this does it:"
  echo "  mkdir -p $ROOT/IntrA && cd $ROOT/IntrA && gdown --folder 1yjLdofRRqyklgwFOC0K4r7ee1LPKstPh"
  echo "The 5-fold split files are in splits/intra/ (committed with this repository)."
}

rbc() {
  echo "== 3D red blood cells (Simionato et al., 2021; 3D masks by Waibel et al., Zenodo 7031924, CC BY 4.0)"
  echo "The point clouds are built locally from the public masks (only the 825 masks are read from the"
  echo "Zenodo archive; the output is identical to the files used in the paper)."
  python "$(dirname "$0")/prepare_rbc.py" --root "$ROOT"
  echo "The 5-fold split files are in splits/rbc/."
}

modelnet40() {
  echo "== ModelNet40 (2048-point HDF5 release used by PointNet++, DGCNN and PointNeXt)"
  mkdir -p "$ROOT/ModelNet40"
  zip="$ROOT/ModelNet40/modelnet40_ply_hdf5_2048.zip"
  stanford="https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
  # copy of the same archive on the Hugging Face Hub, used when the Stanford server is offline
  hub="https://huggingface.co/datasets/Msun/modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip"
  hub_sha256="f01b8189281fae5790e39deb9f3eca86e446b771bdc665c6ad05f28d039b20e7"
  if [ ! -f "$zip" ]; then
    echo "downloading $stanford (about 435 MB)"
    if ! curl -fL --connect-timeout 20 -o "$zip.part" "$stanford"; then
      echo "the Stanford server did not answer; downloading the copy from the Hugging Face Hub"
      curl -fL -o "$zip.part" "$hub"
      echo "$hub_sha256  $zip.part" | sha256sum -c - || { echo "checksum mismatch, removing the download"; rm -f "$zip.part"; return 1; }
    fi
    mv "$zip.part" "$zip"
  fi
  (cd "$ROOT/ModelNet40" && unzip -qo modelnet40_ply_hdf5_2048.zip)
  echo "ModelNet40 ready: $(ls "$ROOT"/ModelNet40/modelnet40_ply_hdf5_2048/*.h5 | wc -l) h5 files"
}

scanobjectnn() {
  echo "== ScanObjectNN (Uy et al., ICCV 2019), hardest variant PB_T50_RS"
  echo "The dataset requires agreeing to the authors' terms of use; request the download link at"
  echo "  https://hkust-vgd.github.io/scanobjectnn/"
  echo "Then unpack h5_files.zip and point the config at the main_split folder:"
  echo "  $ROOT/ScanObjectNN/{training,test}_objectdataset_augmentedrot_scale75.h5"
  echo "(e.g. ln -s /path/to/h5_files/main_split $ROOT/ScanObjectNN)"
}

atlas1() {
  echo "== ATLAS-1 (released with the paper)"
  mkdir -p "$ROOT/ATLAS-1"
  if python -c "import huggingface_hub" 2>/dev/null; then
    python - "$ROOT/ATLAS-1" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Sentinal4D/ATLAS-1", repo_type="dataset", local_dir=sys.argv[1])
print("downloaded ATLAS-1 to", sys.argv[1])
PY
  else
    echo "pip install -e \".[data]\" (for huggingface_hub), then re-run; or download from https://huggingface.co/datasets/Sentinal4D/ATLAS-1"
  fi
  echo "Layout: $ROOT/ATLAS-1/metadata.csv, $ROOT/ATLAS-1/data/*.obj (meshes) and, when present, pointclouds_4096/*.npy"
  echo "(the training point clouds). The train/val/test split is splits/atlas1/split0.csv."
  echo "Without pointclouds_4096 add '--override data.source=mesh' to train.py (points are sampled from the mesh vertices)."
}

case "$what" in
  intra) intra ;;
  rbc) rbc ;;
  modelnet40) modelnet40 ;;
  scanobjectnn) scanobjectnn ;;
  atlas1) atlas1 ;;
  all)
    # one failing dataset (e.g. a missing optional dependency) does not stop the others
    for d in intra rbc modelnet40 scanobjectnn atlas1; do
      ( "$d" ) || echo "!! $d did not complete; see the messages above and re-run: bash scripts/download_data.sh $d"
      echo
    done ;;
  *) echo "unknown dataset '$what'"; exit 1 ;;
esac
