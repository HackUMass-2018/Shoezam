window.addEventListener("load", () => {
    document.getElementById("upload").addEventListener("change", () => {
        document.getElementById("form").submit();
    });
});

window.addEventListener("click", () => {
    document.getElementById("upload").dispatchEvent(new MouseEvent("click"));
});
