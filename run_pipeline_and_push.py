import subprocess
import sys
from datetime import datetime

def run(script, optional=False):
    print(f"\n[RUN] {script}")
    try:
        subprocess.run([sys.executable, script], check=True)
        print(f"[OK] Finished: {script}")
    except subprocess.CalledProcessError:
        if optional:
            print(f"[WARN] Optional script failed or skipped: {script}")
        else:
            print(f"[ERROR] Script failed: {script}")
            sys.exit(1)

def git_push():
    print("\n[INFO] Pushing to GitHub (force)...")
    commit_msg = f"Auto-update: predictions as of {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push", "--force", "origin", "main"], check=True)
        print("[OK] Force push complete.")
    except subprocess.CalledProcessError:
        print("[❌] Git push failed. Manual fix may be required.")


if __name__ == "__main__":
    print("[START] Full boosted innings pipeline...\n")

    run("Scrape_Fan_Graph.py")
    run("get_scores_full.py")
    run("predict_over_4_5.py")
    run("merge_predictions.py")
    run("train_model.py", optional=True)
    run("backfill_predict_over_4_5.py", optional=True)

    git_push()
    print("\n[COMPLETE] All tasks finished.")

