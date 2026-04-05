# Plan #0200: Docker-based Test Execution

- **Date**: 2026-04-04
- **Status**: completed
- **Module**: docker-compose, CLAUDE.md, testing rules

## 目標

統一測試執行方式，讓所有測試一律在 Docker 容器內跑，符合 CLAUDE.md 規定的「所有指令在容器內執行」。

## 影響範圍

- `docker-compose.yml` — 新增 `test` service
- `.claude/CLAUDE.md` — Commands 區塊改為 Docker 版
- `.claude/rules/testing.md` — 新增 Docker 執行方式

## 實作步驟

1. `docker-compose.yml` 新增 `test` service（extends dev，去掉 port，改 container name）
2. 更新 `.claude/CLAUDE.md` 測試/品質指令為 Docker 版
3. 更新 `.claude/rules/testing.md` 加上 Docker 區塊
4. 驗證 `docker compose run --rm test` 跑全套通過

## 完成標準

- `docker compose run --rm test` 成功跑完全套測試
- CLAUDE.md 測試指令全部改為 Docker 版
- testing.md 包含 Docker 執行說明
