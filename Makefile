.PHONY: dev api ui lint eval up down

dev:
	docker-compose up --build

up:
	docker-compose up -d

down:
	docker-compose down

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run ui/app.py

lint:
	ruff check .

eval:
	python scripts/run_eval.py --dataset eval/datasets/sec_qa_v1.jsonl
