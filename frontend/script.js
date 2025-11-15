function askBackend() {

    // Get what the user typed
    var text = document.getElementById("query").value;

    // Send it to backend
    fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ query: text })
    })
    .then(function(response) {
        return response.json();   // Convert reply
    })
    .then(function(data) {
        // Show answer on webpage
        document.getElementById("responseBox").innerText = data.answer;
    });
}
