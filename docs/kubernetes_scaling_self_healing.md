# Local Kubernetes scaling and Pod replacement

This exercise uses the existing Docker Desktop `docker-desktop` context and the
`kuberoute-gateway` Deployment and ClusterIP Service. It changes only the live
Deployment's replica count; [`k8s/gateway.yaml`](../k8s/gateway.yaml) remains at one
replica. Run these commands from PowerShell after confirming the local image is
available to the Docker Desktop Kubernetes node.

## Scale to three

```powershell
kubectl config current-context  # expect docker-desktop
kubectl get deployment/kuberoute-gateway service/kuberoute-gateway
kubectl get pods -l app=kuberoute-gateway
kubectl scale deployment/kuberoute-gateway --replicas=3
kubectl rollout status deployment/kuberoute-gateway --timeout=120s
kubectl get pods -l app=kuberoute-gateway
```

The Deployment should show `3/3` Ready. Check the Service's EndpointSlices rather
than inferring Service membership from the Pod count:

```powershell
$slices = kubectl get endpointslice -l kubernetes.io/service-name=kuberoute-gateway -o json | ConvertFrom-Json
$endpoints = @($slices.items | ForEach-Object { $_.endpoints } | ForEach-Object { $_ })
@($endpoints | Where-Object { $_.conditions.ready -eq $true }).Count  # expect 3
```

For a local HTTP check, run this in a separate PowerShell window:

```powershell
kubectl port-forward --address 127.0.0.1 service/kuberoute-gateway 18000:8000
```

From the first window, check the HTTP status of each endpoint:

```powershell
foreach ($path in @('/health', '/health/live', '/health/ready', '/v1/models')) {
    $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:18000$path"
    "$path $($response.StatusCode)"  # expect HTTP 200
}
```

Stop the port-forward with Ctrl+C before deleting a Pod.
The explicit `--address 127.0.0.1` keeps the temporary listener on the host's
loopback interface.

## Observe automatic replacement

After confirming three Ready Pods, select and delete one Pod normally:

```powershell
$pods = kubectl get pods -l app=kuberoute-gateway -o json | ConvertFrom-Json
if ($pods.items.Count -ne 3) { throw "Expected three gateway Pods" }
$podName = $pods.items[0].metadata.name
kubectl delete pod $podName --wait=false
kubectl get pods -l app=kuberoute-gateway
kubectl rollout status deployment/kuberoute-gateway --timeout=120s
kubectl get pods -l app=kuberoute-gateway
```

The deleted Pod should disappear and a new Pod with a different name should
appear. While it starts, the EndpointSlice can show two ready endpoints and one
not-ready endpoint. Repeat the EndpointSlice check above after rollout; it
should return three ready endpoints again. Do not use force deletion for this
test.

## Restore the baseline

```powershell
kubectl scale deployment/kuberoute-gateway --replicas=1
kubectl rollout status deployment/kuberoute-gateway --timeout=120s
kubectl get pods -l app=kuberoute-gateway
```

Repeat the EndpointSlice check and expect one ready endpoint. The Service
remains ClusterIP-only; this exercise creates no Ingress, NodePort, or
LoadBalancer. Stop any temporary port-forward before finishing.

```powershell
Get-NetTCPConnection -LocalPort 18000 -State Listen -ErrorAction SilentlyContinue
```

This should return no listener after the port-forward has stopped.

## Observed local run (October 20, 2026)

| Stage | Ready Pods / desired | Ready Service endpoints | Observation |
| --- | ---: | ---: | --- |
| Before scaling | 1/1 | 1 | Existing gateway Pod healthy |
| After scaling | 3/3 | 3 | All three Pods Ready |
| Immediately after deleting `kuberoute-gateway-7b685cc54-7wjpb` | 2/3 | 2 | New Pod `kuberoute-gateway-7b685cc54-9vwbw` appeared at 0/1 Ready |
| After replacement became Ready | 3/3 | 3 | Replacement Pod joined the EndpointSlice |
| After cleanup | 1/1 | 1 | Original baseline restored |

With three replicas Ready, `/health`, `/health/live`, `/health/ready`, and
`/v1/models` each returned HTTP 200 through the localhost-bound port-forward.
No LLM inference calls were made. Service endpoint membership and HTTP
availability were verified; per-request load balancing was not measured.
All three replicas ran on the same single Docker Desktop node, so this does
not demonstrate node failure recovery or multi-node high availability.
No failures occurred during the run. The port-forward was stopped and the live
Deployment was restored to one replica with one ready Service endpoint.
