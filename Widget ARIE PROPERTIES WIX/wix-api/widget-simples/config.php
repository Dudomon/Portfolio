<?php
/**
 * Arquivo de configuração para o webhook Sienge->Wix
 */

return [
    // Configurações da API Wix
    'wix' => [
        'api_key' => 'IST.eyJraWQiOiJQb3pIX2FDMiIsImFsZyI6IlJTMjU2In0.eyJkYXRhIjoie1wiaWRcIjpcImVjMjE0MzNmLTVlZjMtNDIxMC1iNDBkLTBmYTQzOWM4MzM4YlwiLFwiaWRlbnRpdHlcIjp7XCJ0eXBlXCI6XCJhcHBsaWNhdGlvblwiLFwiaWRcIjpcImU0MjJjYzdjLWY2MTQtNGEzMC05ZTU5LWQ2ZmM3ODU4ZWFiN1wifSxcInRlbmFudFwiOntcInR5cGVcIjpcImFjY291bnRcIixcImlkXCI6XCI2MThlNjEzOS03NTRlLTQ0ZmMtYThhMi0zMGFhYTY5NzgwNWVcIn19IiwiaWF0IjoxNzU0MjY4MzA2fQ.JwWU8ub7hAxs_dWuXoKl1LrMNtEazRnwQwEqwo6ELmtjslaZYoDPxleR6dQLwNfS11qqF-58n6xdBHCREfKp7-KnSC3BHlTYAL6QH0lj8QcrIJykdUPn1VIcq5IvOSevG8-0q2k9zooAxq_rHWJNmnTYP4guVKsWAuOXBiwHVyptFTQ1WEatIdWrcP9VeugxybbpYhxir3XQEpvmUOYvZKReg79ZOpbZrXKiiiFp8rjp0vteAyuwvXrTR-aY9lMTEck_P4n62wgdqMZ2IBDjnxWkBChoP6d5mXmLBN_5_-bp7sJaWz4VU7NDbWfUSTpNwPiwSgjdA_gbZ7CYmJm7BA',
        'site_id' => '96359d97-c440-4f99-95be-77ba2f71c476',
        'collection_id' => 'Cliente',    // ID da coleção no Wix
        'api_base_url' => 'https://www.wixapis.com/wix-data/v1/collections'
    ],
    
    // Configurações do proxy atual
    'proxy' => [
        'url' => 'https://www.radioentrerios.com.br/wp-content/backends/cors-proxy.php?url=',
        'enabled' => true
    ],
    
    // Configurações da API Sienge
    'sienge' => [
        'base_url' => 'https://api.sienge.com.br/arieproperties/public/api/v1',
        'test_cpf' => '37455407866',
        'test_name' => 'THAIS CRISTINA JULIO BASTOS'
    ],
    
    // Configurações de senha
    'password' => [
        'length' => 8,
        'characters' => 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    ],
    
    // Configurações de email
    'email' => [
        'from_name' => 'Arie Properties',
        'from_email' => 'noreply@arieproperties.com',
        'subject' => 'Bem-vindo à Arie Properties - Seus dados de acesso',
        'portal_url' => 'https://seusite.wixsite.com/portal-cliente' // Substitua pela URL do seu portal
    ],
    
    // Configurações de log
    'logging' => [
        'enabled' => true,
        'file' => __DIR__ . '/webhook-logs.txt',
        'max_size' => 10485760 // 10MB
    ],
    
    // Configurações de segurança
    'security' => [
        'webhook_secret' => '', // Token secreto para validar webhooks (opcional)
        'allowed_ips' => []     // IPs permitidos para chamar o webhook (opcional)
    ]
];
?>