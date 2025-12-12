<?php

declare(strict_types=1);

namespace LegalDocAnalyzer\Services;

use Firebase\JWT\JWT;
use Firebase\JWT\Key;
use LegalDocAnalyzer\Exceptions\AuthenticationException;

class AuthService
{
    private string $secret;
    private string $algorithm;
    private int $expiration;

    public function __construct()
    {
        $this->secret = $_ENV['JWT_SECRET'];
        $this->algorithm = $_ENV['JWT_ALGORITHM'];
        $this->expiration = (int) $_ENV['JWT_EXPIRATION'];
    }

    public function generateToken(array $payload): string
    {
        $issuedAt = time();
        $expiresAt = $issuedAt + $this->expiration;

        $claims = array_merge($payload, [
            'iat' => $issuedAt,
            'exp' => $expiresAt,
            'iss' => 'legal-doc-analyzer',
        ]);

        return JWT::encode($claims, $this->secret, $this->algorithm);
    }

    public function validateToken(string $token): array
    {
        try {
            $decoded = JWT::decode($token, new Key($this->secret, $this->algorithm));
            return (array) $decoded;

        } catch (\Exception $e) {
            throw new AuthenticationException('Invalid or expired token', 401, $e);
        }
    }

    public function refreshToken(string $token): string
    {
        $payload = $this->validateToken($token);

        unset($payload['iat'], $payload['exp']);

        return $this->generateToken($payload);
    }

    public function authenticate(string $username, string $password): ?array
    {
        $user = $this->getUserByUsername($username);

        if (!$user || !password_verify($password, $user['password_hash'])) {
            return null;
        }

        return [
            'id' => $user['id'],
            'username' => $user['username'],
            'email' => $user['email'],
            'roles' => $user['roles'] ?? ['user'],
        ];
    }

    private function getUserByUsername(string $username): ?array
    {
        return [
            'id' => '123e4567-e89b-12d3-a456-426614174000',
            'username' => 'demo',
            'email' => 'demo@example.com',
            'password_hash' => password_hash('demo123', PASSWORD_BCRYPT),
            'roles' => ['admin', 'analyst'],
        ];
    }

    public function hasPermission(array $userRoles, string $requiredPermission): bool
    {
        $rolePermissions = [
            'admin' => ['*'],
            'analyst' => ['documents:read', 'documents:write', 'analysis:read', 'analysis:write'],
            'viewer' => ['documents:read', 'analysis:read'],
        ];

        foreach ($userRoles as $role) {
            $permissions = $rolePermissions[$role] ?? [];

            if (in_array('*', $permissions) || in_array($requiredPermission, $permissions)) {
                return true;
            }
        }

        return false;
    }
}
