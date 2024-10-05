function calculateGoal() {
    const amount = document.getElementById('goalAmount').value;
    const time = document.getElementById('timePeriod').value;
    const salary = document.getElementById('salary').value;

    // Simple Calculation Example
    const monthlySaving = (amount / (time * 12)).toFixed(2);

    document.getElementById('goalResult').innerText = `You need to save ₹${monthlySaving} per month to achieve your goal.`;
}
