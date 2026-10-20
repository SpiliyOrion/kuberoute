# Local Prometheus monitoring

This setup runs one Prometheus Pod beside KubeRoute in Docker Desktop Kubernetes'
`default` namespace. It discovers gateway Pods by the existing `app=kuberoute-gateway`
label and named `http` container port, then scrapes each Pod's `/metrics` every 15
seconds. The Prometheus and gateway Services remain ClusterIP-only. The [manifest](../k8s/prometheus.yaml)
does not contain credentials, public listeners, or a persistent volume.
These manifests target `default` to match the existing gateway; if you deploy
the gateway elsewhere, change the namespace on all monitoring resources and
the RoleBinding's ServiceAccount subject together.

Prometheus uses the official, pinned
[`quay.io/prometheus/prometheus:v3.13.4`](https://github.com/prometheus/prometheus/releases/tag/v3.13.4)
image. Kubernetes will download it from Quay.io if it is not already in the local
image store. If registry access is restricted, the Pod will show `ImagePullBackOff`;
resolve registry access through your normal environment before retrying. This guide
does not require installing Prometheus or other software on Windows.

## Deploy from PowerShell

Run these commands from the repository root. The gateway manifest uses the local
`kuberoute-gateway:latest` image with `imagePullPolicy: Never`; rebuild and restart
the gateway if its running Pod predates the `/metrics` endpoint. The restart keeps
the existing Deployment, Service, and environment configuration.

```powershell
kubectl config current-context  # expect docker-desktop
kubectl get deployment/kuberoute-gateway service/kuberoute-gateway -n default
docker build -t kuberoute-gateway:latest .
kubectl rollout restart deployment/kuberoute-gateway -n default
kubectl rollout status deployment/kuberoute-gateway -n default --timeout=120s
kubectl apply --dry-run=server --validate=strict -f k8s/prometheus.yaml
kubectl apply -f k8s/prometheus.yaml
kubectl rollout status deployment/kuberoute-prometheus -n default --timeout=180s
kubectl get deployment,service,pods -n default -l app=kuberoute-prometheus
kubectl exec deployment/kuberoute-prometheus -n default -- promtool check config /etc/prometheus/prometheus.yml
```

The Deployment runs one non-root Pod with a read-only root filesystem, dropped
capabilities, and an ephemeral `/prometheus` data directory. It requests
100m CPU/256Mi memory, limits usage to 500m CPU/512Mi memory, and retains data
for up to six hours or 256 MB. The ephemeral volume has a 512Mi size limit;
data is lost when the Pod is replaced. The Service exposes only port 9090 inside
the cluster. The ServiceAccount can list and watch Pods only in `default`; it has
no permission to read Secrets or Pods in other namespaces.

## Verify scraping and query metrics

In one PowerShell window, start the Prometheus port-forward. The explicit
address binds it only to the host loopback interface:

```powershell
kubectl port-forward --address 127.0.0.1 service/kuberoute-prometheus 9090:9090
```

In another window, verify readiness, open `http://127.0.0.1:9090/` in a browser,
and inspect the gateway targets. Each Ready gateway Pod should appear as a
separate UP target. Prometheus sets `instance` to the Pod name instead of
retaining a Pod IP as a metric label.

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:9090/-/ready
$targets = (Invoke-RestMethod 'http://127.0.0.1:9090/api/v1/targets?state=active').data.activeTargets
$targets | Where-Object { $_.labels.job -eq 'kuberoute-gateway' } | ForEach-Object { "$($_.labels.instance): $($_.health)" }
```

To generate traffic without an LLM, use a second localhost-only port-forward
in a third window, then request the health endpoint from PowerShell:

```powershell
kubectl port-forward --address 127.0.0.1 service/kuberoute-gateway 18000:8000
```

```powershell
1..6 | ForEach-Object { Invoke-RestMethod http://127.0.0.1:18000/health | Out-Null }
Start-Sleep -Seconds 20
1..6 | ForEach-Object { Invoke-RestMethod http://127.0.0.1:18000/health | Out-Null }
Start-Sleep -Seconds 20

function Get-PromQuery([string]$expression) {
    $encoded = [uri]::EscapeDataString($expression)
    (Invoke-RestMethod "http://127.0.0.1:9090/api/v1/query?query=$encoded").data.result
}

Get-PromQuery 'sum(kuberoute_http_requests_total{job="kuberoute-gateway",route="/health",status_code="200"})'
Get-PromQuery 'sum(rate(kuberoute_http_requests_total{job="kuberoute-gateway",route="/health",status_code="200"}[1m]))'
Get-PromQuery 'sum(kuberoute_http_request_duration_seconds_count{job="kuberoute-gateway",route="/health",status_code="200"})'
Get-PromQuery 'sum(rate(kuberoute_http_request_duration_seconds_sum{job="kuberoute-gateway",route="/health",status_code="200"}[1m])) / sum(rate(kuberoute_http_request_duration_seconds_count{job="kuberoute-gateway",route="/health",status_code="200"}[1m]))'
```

The queries return total successful `/health` requests, requests per second,
histogram observations, and mean request duration in seconds. Rates require at
least two scrapes in the one-minute window. In the local validation, 12 health
requests produced a counter and histogram count of 12, a rate of about 0.237
requests/second, and a mean observed duration of about 0.00068 seconds; timing
values vary. `/health` does not run inference, so this health-endpoint latency
does not represent LLM inference latency. Scaling the gateway to two Pods
briefly produced two distinct UP targets, then it was restored to one replica.
This checks independent target discovery, not per-request load balancing.

Stop both port-forwards with Ctrl+C when finished. Verify no local listener
remains on ports 9090 or 18000:

```powershell
Get-NetTCPConnection -LocalPort 9090,18000 -State Listen -ErrorAction SilentlyContinue
```

## Troubleshoot or remove

If the Prometheus Pod is not Ready, inspect its status and events with
`kubectl describe pod -n default -l app=kuberoute-prometheus`. If a gateway
target is DOWN, check that the gateway's `/metrics` returns HTTP 200 through
the localhost port-forward and that the running gateway image includes Phase 7.
Check Prometheus discovery permissions with:

```powershell
kubectl auth can-i list pods --as=system:serviceaccount:default:kuberoute-prometheus -n default
kubectl auth can-i watch pods --as=system:serviceaccount:default:kuberoute-prometheus -n default
kubectl auth can-i list secrets --as=system:serviceaccount:default:kuberoute-prometheus -n default
```

The expected answers are `yes`, `yes`, and `no`. The gateway exports only bounded
HTTP route, method, and status labels; Prometheus adds a fixed job and a Pod-name
instance label. It does not scrape prompts, generated responses, API keys, or
request bodies. The web UI has no authentication, so keep the Service private
and use only the localhost-bound port-forward for local access.

To remove only the monitoring resources after stopping the port-forward:

```powershell
kubectl delete -f k8s/prometheus.yaml
```
