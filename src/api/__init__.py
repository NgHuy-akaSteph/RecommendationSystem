"""Tầng serving — biến mô hình offline thành API gợi ý dùng được (demo, API HTTP).

Module này **không** import `src.eval.*`: evaluator chấm điểm trên nhãn test, còn tầng
serving chỉ biết `TrainData` và hợp đồng `score_users`.
"""
