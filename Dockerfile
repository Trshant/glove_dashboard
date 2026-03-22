FROM golang:1.22-alpine AS build
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o server cmd/server/main.go

FROM alpine:3.19
WORKDIR /app
COPY --from=build /app/server .
COPY --from=build /app/templates ./templates
COPY --from=build /app/case_data ./case_data
EXPOSE 8001
ENTRYPOINT ["./server"]
