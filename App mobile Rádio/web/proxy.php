<?php
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');
header('Content-Type: application/json');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    exit();
}

$url = 'https://radioentrerios.com.br/wp-content/noticias/get_noticias.php';
$params = [];
if (isset($_GET['limit'])) {
    $params[] = 'limit=' . intval($_GET['limit']);
}
if (isset($_GET['offset'])) {
    $params[] = 'offset=' . intval($_GET['offset']);
}
if (!empty($params)) {
    $url .= '?' . implode('&', $params);
}

$response = file_get_contents($url);
echo $response;
?>