# The community server is source-only; build the pinned upstream release.
FROM golang:1.26.10-alpine3.23 AS build
ENV CGO_ENABLED=0 GOTOOLCHAIN=local
RUN go install github.com/minio/minio@RELEASE.2025-10-15T17-29-55Z
RUN cp /go/pkg/mod/github.com/minio/minio@*/LICENSE /tmp/MINIO-LICENSE

FROM alpine:3.23.2
RUN apk add --no-cache ca-certificates \
    && addgroup -g 1000 minio \
    && adduser -D -u 1000 -G minio minio \
    && mkdir /data && chown minio:minio /data
COPY --from=build /go/bin/minio /usr/local/bin/minio
COPY --from=build /tmp/MINIO-LICENSE /licenses/MINIO-LICENSE
USER 1000:1000
EXPOSE 9000
ENTRYPOINT ["minio"]
