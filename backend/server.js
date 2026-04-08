const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());

app.post('/predict', (req, res) => {
    const inputData = JSON.stringify(req.body);

    const parentDir = path.join(__dirname, '..');

    const scriptPath = path.join(parentDir, 'predict.py');

    const pythonProcess = spawn('python', [scriptPath, inputData], { cwd: parentDir });

    let dataString = '';
    let errorString = '';


    pythonProcess.stdout.on('data', (data) => {
        dataString += data.toString();
    });


    pythonProcess.stderr.on('data', (data) => {
        errorString += data.toString();
        console.error(`Python Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            return res.status(500).json({ 
                message: "Python script failed to run", 
                details: errorString 
            });
        }

        try {
            const result = JSON.parse(dataString);
            if (result.error) {
                return res.status(500).json({ message: result.error });
            }
            res.json(result);
        } catch (error) {
            console.error("Raw unparseable output from Python:", dataString);
            res.status(500).json({ 
                message: "Failed to parse prediction output", 
                raw_output: dataString 
            });
        }
    });
});

app.listen(PORT, () => {
    console.log(`Express server running on http://localhost:${PORT}`);
});