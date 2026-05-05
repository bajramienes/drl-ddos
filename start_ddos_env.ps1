# start_ddos_env.ps1
# PowerShell script to create a simple DDoS test environment in Docker
# Run with:  PowerShell -ExecutionPolicy Bypass -File .\start_ddos_env.ps1

Write-Host "=== Starting DDoS Test Environment ==="

# 1. Create docker network (if not exists)
if (-not (docker network ls --format "{{.Name}}" | Select-String -Pattern "^ddosnet$")) {
    Write-Host "Creating network 'ddosnet'..."
    docker network create ddosnet
} else {
    Write-Host "Network 'ddosnet' already exists."
}

# 2. Start target container
if (-not (docker ps -a --format "{{.Names}}" | Select-String -Pattern "^ddos_target$")) {
    Write-Host "Starting target container..."
    docker run -d --name ddos_target --network ddosnet nginx
} else {
    Write-Host "Target container already exists, starting it..."
    docker start ddos_target
}

# 3. Start attacker container
if (-not (docker ps -a --format "{{.Names}}" | Select-String -Pattern "^ddos_attacker$")) {
    Write-Host "Starting attacker container..."
    docker run -d --name ddos_attacker --network ddosnet alpine sh -c "while true; do wget -q -O- http://ddos_target; done"
} else {
    Write-Host "Attacker container already exists, starting it..."
    docker start ddos_attacker
}

# 4. Start monitor container
if (-not (docker ps -a --format "{{.Names}}" | Select-String -Pattern "^ddos_monitor$")) {
    Write-Host "Starting monitor container..."
    docker run -d --name ddos_monitor --network ddosnet alpine sh -c "while true; do wget -q -O- http://ddos_target; sleep 2; done"
} else {
    Write-Host "Monitor container already exists, starting it..."
    docker start ddos_monitor
}

# 5. Show running containers
Write-Host "=== Running containers ==="
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"

Write-Host "DDoS environment is ready!"
