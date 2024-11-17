// Load experiment data from JSON file
document.addEventListener('DOMContentLoaded', async () => {
    try {
      const response = await fetch('./mlruns/experiments.json');
      const data = await response.json();

      // Extract experiments array
      const experiment = data.metrics;
      const tags = data.tags;
  
      // Get HTML element
      const tbody = document.getElementById('experiments-tbody');
      const paragraph = document.getElementById('run-name');

      // Populate table with the first experiment
      if (Object.keys(experiment).length > 0) {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td>${tags['mlflow.source.name']}</td>
          <td>${experiment.eval_loss}</td>
          <td>${experiment.eval_accuracy}</td>
        `;
        tbody.appendChild(row);
      }

      // Get paragraph element
      if (Object.keys(tags).length > 0) {
        paragraph.innerHTML = tags['mlflow.runName'];
      }

    } catch (error) {
      console.error('Error loading experiments:', error);
    }
  });