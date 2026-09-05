TRACKED_GO_FILES := $(shell git ls-files '*.go')

.PHONY: format help lint security test test-units

format: ## Format tracked Go source files.
	go tool goimports -w $(TRACKED_GO_FILES)
	go tool gofumpt -w $(TRACKED_GO_FILES)

help: ## List available targets.
	@awk 'BEGIN { FS = ":.*##" } /^[a-zA-Z0-9_-]+:.*##/ { printf "%-16s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

lint: ## Run static analysis and spelling checks.
	go vet ./...
	go tool staticcheck ./...
	go tool ineffassign ./...
	go tool errcheck ./...
	go tool go-errorlint -errorf ./...
	go tool exhaustive ./...
	go tool misspell -locale US -source=go $(TRACKED_GO_FILES)
	typos --locale en-us

security: ## Check for Go security defects and reachable vulnerabilities.
	go tool gosec ./...
	go tool govulncheck ./...

test: ## Run every test suite.
	$(MAKE) test-units

test-units: ## Run unit and collaboration tests.
	go test ./...
