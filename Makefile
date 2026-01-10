.PHONY: install dev-backend dev-desktop dev-web

install:
	npm install
	cd services/backend && pip install -r requirements.txt

dev-backend:
	cd services/backend && uvicorn main:app --reload

dev-desktop:
	npm run electron:dev -w apps/desktop

dev-web:
	npm run dev -w apps/web
