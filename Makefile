.PHONY: build run test train-create train-add train-import clean

# Build binaries
build:
	go build -o bin/server cmd/server/main.go
	go build -o bin/train cmd/train/main.go

# Run the web server
run:
	go run cmd/server/main.go

# Run tests
test:
	go test ./...

# Training commands
train-create:
	go run cmd/train/main.go create --input $(INPUT) --output $(OUTPUT) --min-count $(or $(MIN_COUNT),10)

train-add:
	go run cmd/train/main.go add --model $(MODEL) --input $(INPUT)

train-import:
	go run cmd/train/main.go import --input $(INPUT) --output $(OUTPUT)

# Clean build artifacts
clean:
	rm -rf bin/
