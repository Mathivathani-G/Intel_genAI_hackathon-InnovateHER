// Sample data for previous history (replace this with real-time data from backend)
const previousHistory = [500, 700, 650, 800, 600];

// Function to fetch prediction from the backend
async function fetchPrediction() {
  const response = await fetch('/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ history: previousHistory }),
  });
  const data = await response.json();
  return data;
}

// Budget vs Expense Graph
async function renderBudgetExpenseChart() {
  const prediction = await fetchPrediction();
  
  const ctx = document.getElementById('budgetExpenseChart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Previous Expenses', 'Predicted Budget'],
      datasets: [{
        label: 'Amount in $',
        data: [previousHistory.reduce((a, b) => a + b, 0), prediction.budget],
        backgroundColor: ['#77b748', '#4e9f3d'],
      }]
    },
    options: {
      scales: {
        y: {
          beginAtZero: true
        }
      }
    }
  });
}

// Initialize the dashboard chart
renderBudgetExpenseChart();
