function calculateCustomGoal() {
    const name = document.getElementById('customGoalName').value;
    const amount = document.getElementById('goalAmount').value;
    const time = document.getElementById('timePeriod').value;
    const salary = document.getElementById('salary').value;

    // Simple Calculation Example
    const monthlySaving = (amount / (time * 12)).toFixed(2);

    document.getElementById('customGoalResult').innerText = `For your goal "${name}", you need to save ₹${monthlySaving} per month.`;
}
