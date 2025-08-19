# cleanup_ddos_env.ps1
# Stop and remove containers and the ddosnet network

$containers = @('ddos_target','ddos_attacker','ddos_monitor')
$network = 'ddosnet'

Write-Host "=== CLEANUP: DDoS Test Environment ==="
$confirm = Read-Host "This will STOP and REMOVE containers and the '$network' network. Type 'YES' to continue"
if ($confirm -ne 'YES') {
  Write-Host "Aborted."
  exit 0
}

# Stop containers if present
foreach ($c in $containers) {
  if (docker ps -a --format "{{.Names}}" | Select-String -Pattern "^$c$") {
    Write-Host "Stopping $c ..."
    docker stop $c | Out-Null
  } else {
    Write-Host "$c not found, skipping stop."
  }
}

# Remove containers if present
foreach ($c in $containers) {
  if (docker ps -a --format "{{.Names}}" | Select-String -Pattern "^$c$") {
    Write-Host "Removing $c ..."
    docker rm $c | Out-Null
  } else {
    Write-Host "$c not found, skipping remove."
  }
}

# Remove network if present
if (docker network ls --format "{{.Name}}" | Select-String -Pattern "^$network$") {
  Write-Host "Removing network '$network' ..."
  docker network rm $network | Out-Null
} else {
  Write-Host "Network '$network' not found, skipping."
}

Write-Host "Remaining containers:"
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
Write-Host "Remaining networks:"
docker network ls --format "table {{.Name}}\t{{.Driver}}"

Write-Host "Cleanup complete."
