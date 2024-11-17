## Documentation

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MLflow Experiments</title>
    <link rel="stylesheet" href="../../../docs/static/css/mlflow_docs.css">
    <h4> MLFlow Training Summary</h4>
    <p> Placeholder section to include summary details on the training job run. The metrics below will reflect the latest successful run.</p>
</head>
<body>
    <div>
    <p class="run-summary-style" id="run-name"></p>
    </div>
    <div class="container">
        <details>
            <summary>Metrics</summary>
            <table id="experiments-table">
                <thead>
                    <tr>
                        <th>Source</th>
                        <th>Eval Loss</th>
                        <th>Eval Accuracy</th>
                    </tr>
                </thead>
                <tbody id="experiments-tbody"></tbody>
            </table>
        </details>
        <script type="text/javascript" src="../../../docs/static/js/mlflow_docs.js"></script>
    </div>
</body>
</html>