// Wait till the browser is ready to render the game (avoids glitches)
window.requestAnimationFrame(function () {
    // Create game manager and expose globally
    window.gm = new GameManager(4, KeyboardInputManager, HTMLActuator, LocalStorageManager);

    // Optional: helper to get a clean 4x4 numeric board
    window.getBoard = function() {
        let grid = window.gm.serialize().grid;
        let out = Array(16).fill(0);
        for (let r = 0; r < 4; r++) {
            for (let c = 0; c < 4; c++) {
                let tile = grid.cells[r][c];
                if (tile && typeof tile.value === "number") {
                    out[4*c+r] = tile.value;
                }
            }
        }
        return out;
    };

    // Optional: print board nicely in console
    window.printBoard = function() {
        console.table(window.gm.serialize());
    };
});
