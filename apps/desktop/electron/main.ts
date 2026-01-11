import { app, BrowserWindow } from 'electron';
import path from 'path';

let mainWindow: BrowserWindow | null;

const createMainWindow = () => {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        backgroundColor: '#001219',
        autoHideMenuBar: true, // Hide default menu bar
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            webSecurity: true,
            allowRunningInsecureContent: false,
        },
    });

    mainWindow.setMenu(null); // Explicitly remove the menu

    const appURL = app.isPackaged
        ? `file://${path.join(__dirname, '../out/index.html')}`
        : 'http://localhost:3001';

    mainWindow.loadURL(appURL);

    mainWindow.webContents.on('did-fail-load', (_, errorCode, errorDescription) => {
        console.error('FAILED TO LOAD:', errorCode, errorDescription);
    });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
};

// Disable hardware acceleration to prevent black screen issues
app.disableHardwareAcceleration();

app.on('ready', () => {
    createMainWindow();
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (mainWindow === null) {
        createMainWindow();
    }
});
