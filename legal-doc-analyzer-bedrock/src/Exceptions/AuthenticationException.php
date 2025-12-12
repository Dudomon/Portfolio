<?php

declare(strict_types=1);

namespace LegalDocAnalyzer\Exceptions;

class AuthenticationException extends \Exception
{
    public function getStatusCode(): int
    {
        return $this->code ?: 401;
    }
}
