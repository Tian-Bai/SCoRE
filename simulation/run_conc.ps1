$cases = 1, 2, 3
$settings = 1, 2

foreach ($i in $cases) {
    foreach ($j in $settings) {
        Write-Host "========================================="
        Write-Host "Running python mdr_conc.py $i $j 0.1 20 0.95 10 1000 1000 100 0"
        Write-Host "========================================="
        python mdr_conc.py $i $j 0.1 20 0.95 10 1000 100 100 0
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "An error occurred with case $i setting $j. Terminating script."
            exit $LASTEXITCODE
        }
    }
}

Write-Host "All simulations completed successfully!"
