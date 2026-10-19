from evaluation import get_performance_stats
import json

DB_PATH = "anpr_history.db"

def run_evaluation():
    print("\n" + "="*40)
    print(" ANPR SYSTEM EVALUATION (Metrics Report)")
    print("="*40)
    
    stats = get_performance_stats(DB_PATH)
    
    if not stats or stats.get("total_detections") == 0:
        print("\n[!] No data found in anpr_history.db.")
        print("Please upload and validate some images first.\n")
        return

    print(f"\n[+] Total Detections:   {stats['total_detections']}")
    print(f"[+] Total Validated:     {stats['validated']}")
    print(f"{'-'*40}")
    print(f"[>] True Positives (TP): {stats['tp']}")
    print(f"[>] False Positives (FP):{stats['fp']}")
    print(f"{'-'*40}")
    print(f"\n[*] FINAL METRICS:")
    print(f"    - Accuracy:  {stats['accuracy']*100:.2f}%")
    print(f"    - Precision: {stats['precision']*100:.2f}%")
    print(f"    - Recall:    {stats['recall']*100:.2f}% (Constant)")
    print(f"\n{'='*40}\n")

if __name__ == "__main__":
    run_evaluation()
