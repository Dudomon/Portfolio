import { fetch } from 'wix-fetch';

export async function fazerLoginExterno(emailCpf, senha) {
    try {
        console.log('Backend WEB: fazendo login para', emailCpf);
        
        const dadosLogin = { senha };
        
        if (emailCpf.includes('@')) {
            dadosLogin.email = emailCpf;
        } else {
            dadosLogin.cpfCnpj = emailCpf;
        }

        const response = await fetch('https://www.arieproperties.com/_functions/login', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(dadosLogin)
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const resultado = await response.json();
        return resultado;
        
    } catch (error) {
        return { 
            success: false, 
            error: error.message || 'Erro de conexão' 
        };
    }
}