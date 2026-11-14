# results/rerun_superseded — bản cũ, khoá chỉ-đọc (chuyển 2026-11-09)

Giữ để đối chiếu / quay lại khi cần. KHÔNG dùng làm kết quả chính, KHÔNG để lại trong `results/rerun/` (merge sẽ báo trùng).

- rerun/main.json            : Sparkov 20% test (SHAP trên mẫu 20%), 28/10. Trùng byte với sample20_reference/sparkov_20pct.json.
- rerun/ulb/main.json        : ULB bản cũ (trước khi bỏ 1.081 dòng trùng; test 56.961 dòng), 5/11.
- rerun/colab_full_{cpu,gpu,rf}.json : bản Colab 8/11 (Gemini) — lẫn CatBoost GPU + SHAP Approximate. Bản mới nhất nằm trên Drive CIES_colab/results_rerun.
- summaries/                 : cies_rerun_summary*(.csv/.json) cũ (Sparkov 20%, ULB cũ). Merge sẽ ghi đè bản trong results/.
- provenance/                : bao_cao_bo_trung.json cũ (ULB).
- MANIFEST.sha256            : checksum từng file.
Kiểm: sha256sum -c MANIFEST.sha256
