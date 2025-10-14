import json
from pathlib import Path
import re

NB_PATH = Path('EV-Battery-Parking-Degradation-Mitigation/train/ranking_predict.ipynb')

PARAM_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# パラメータ（必要に応じて調整してください）\n",
        "LONG_PARK_THRESHOLD_MIN = 360  # 長時間放置の定義（分）\n",
        "CAND_TOP_N_PER_VEHICLE = 10   # 車両別候補 Top-N\n",
        "CAND_GLOBAL_TOP_N = 20        # 全体候補 Top-N（補完用）\n",
        "NEG_SAMPLE_K = 10             # 学習用の負例サンプリング K（0/負なら無効）\n",
        "HOUR_BIN_SIZE = 3             # 時刻ビン幅（h）。例: 3→ 0-2,3-5,... の 3h ビン\n",
        "ALPHA_SMOOTH = 1.0            # ラプラス平滑化の α\n",
        "AG_PRESETS = 'medium_quality_faster_train'  # AutoGluon のプリセット\n",
        "TIME_LIMIT = 300              # 学習の時間制限（秒）\n",
    ],
}


def patch_notebook(nb):
    # 1) パラメータセルを、最初のインポートセルの直後に挿入
    insert_at = None
    for i, c in enumerate(nb.get('cells', [])):
        if c.get('cell_type') == 'code' and any('autogluon.tabular' in s for s in c.get('source', [])):
            insert_at = i + 1
            break
    if insert_at is not None:
        nb['cells'].insert(insert_at, PARAM_CELL)

    # 2) 文字列置換（各セルに対して）
    patterns = [
        (
            r"prepare_sessions\(sessions,\s*long_park_threshold_minutes=360\)",
            r"prepare_sessions(sessions, long_park_threshold_minutes=LONG_PARK_THRESHOLD_MIN)",
        ),
        (
            r"build_candidate_pool_per_vehicle\(sessions,\s*top_n_per_vehicle=10,\s*global_top_n=20\)",
            r"build_candidate_pool_per_vehicle(sessions, top_n_per_vehicle=CAND_TOP_N_PER_VEHICLE, global_top_n=CAND_GLOBAL_TOP_N)",
        ),
        (
            r"build_ranking_training_data\(\n(.*?)negative_sample_k=10\n\s*\)",
            r"build_ranking_training_data(\n\1negative_sample_k=NEG_SAMPLE_K,\n    hour_bin_size=HOUR_BIN_SIZE, alpha_smooth=ALPHA_SMOOTH\n)",
        ),
        (
            r"time_limit=300",
            r"time_limit=TIME_LIMIT",
        ),
        (
            r"presets='medium_quality_faster_train'",
            r"presets=AG_PRESETS",
        ),
    ]

    for c in nb.get('cells', []):
        if c.get('cell_type') != 'code':
            continue
        src = ''.join(c.get('source', []))
        for pat, repl in patterns:
            src = re.sub(pat, repl, src, flags=re.DOTALL)
        c['source'] = [s + ('\n' if not s.endswith('\n') else '') for s in src.split('\n') if s != '']

    return nb


def main():
    nb = json.loads(NB_PATH.read_text(encoding='utf-8'))
    nb2 = patch_notebook(nb)
    NB_PATH.write_text(json.dumps(nb2, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()

