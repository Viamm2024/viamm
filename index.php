<?php 


// Get the current directory of the PHP script
$scriptDirectory = __DIR__;

// Path to the Python script (same directory as the PHP script)
$pythonScriptPath = $scriptDirectory . '/Tester.py';

// Execute the Python script and capture the output
$output = shell_exec("python3 $pythonScriptPath 2>&1");
?>