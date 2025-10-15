import pandas as pd

# Parquet 파일 열기
df = pd.read_parquet("/home/yaaaaaaaang/DeMo/submission/single_agent_2025-10-04-16-55.parquet")

# CSV로 저장
df.to_csv("single_agent_2025-10-04-16-55.csv", index=False)

print("변환 완료: single_agent_2025-10-04-16-55.csv")
