<?php
$logFile = __DIR__ . '/webhook-logs.txt';
if (file_exists($logFile)) {
    echo '<pre>' . htmlspecialchars(tail($logFile, 20)) . '</pre>';
} else {
    echo 'Log file not found';
}

function tail($filename, $lines = 10) {
    $data = file($filename);
    return implode('', array_slice($data, -$lines));
}
?>