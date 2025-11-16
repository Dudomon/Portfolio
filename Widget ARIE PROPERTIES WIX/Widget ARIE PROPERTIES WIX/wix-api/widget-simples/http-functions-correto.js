// http-functions.js - Formato correto para Wix
import {ok, serverError, badRequest} from 'wix-http-functions';
import {fetch} from 'wix-fetch';
import wixData from 'wix-data';
import {send} from 'wix-email';

// Configurações da API Sienge
const API_CONFIG = {
    baseURL: 'https://api.sienge.com.br/arieproperties/public/api/v1',
    auth: {
        username: 'arieproperties-ti',
        password: 'eeoTCIXR9NtEdv118V1L16xZWZR64W7p'
    }
};

// Função de teste simples
export function get_teste(request) {
    return ok({
        headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        body: {
            success: true,
            message: 'Backend funcionando!',
            timestamp: new Date().toISOString()
        }
    });
}

// Função para buscar boletos
export function get_boletosDetalhes(request) {
    try {
        const {cpfCnpj} = request.query;
        
        if (!cpfCnpj) {
            return badRequest({
                body: {
                    success: false,
                    error: 'CPF/CNPJ é obrigatório'
                }
            });
        }

        // Buscar saldo devedor atual
        return fetch(`${API_CONFIG.baseURL}/current-debit-balance?cpf=${cpfCnpj.replace(/\D/g, '')}`, {
            method: 'GET',
            headers: {
                'Authorization': `Basic ${Buffer.from(`${API_CONFIG.auth.username}:${API_CONFIG.auth.password}`).toString('base64')}`,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Erro na API: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Processar dados
            const boletosComDetalhes = [];
            
            for (const bill of data.results || []) {
                const billReceivableId = bill.billReceivableId;
                const tiposParcelas = ['dueInstallments', 'payableInstallments', 'paidInstallments'];
                
                for (const tipo of tiposParcelas) {
                    for (const installment of bill[tipo] || []) {
                        if (installment.generatedBoleto) {
                            boletosComDetalhes.push({
                                numero: installment.installmentNumber || 'N/A',
                                valorOriginal: installment.originalValue || 0,
                                valorAtual: installment.currentBalance || 0,
                                dataVencimento: installment.dueDate,
                                situacao: installment.currentBalance > 0 ? (tipo === 'dueInstallments' ? 'Vencido' : 'A Vencer') : 'Pago',
                                billReceivableId: billReceivableId,
                                installmentId: installment.installmentId,
                                podeSegundaVia: installment.generatedBoleto && installment.currentBalance > 0,
                                nossoNumero: installment.ourNumber || 'N/A'
                            });
                        }
                    }
                }
            }
            
            return ok({
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                body: {
                    success: true,
                    boletos: boletosComDetalhes,
                    total: boletosComDetalhes.length
                }
            });
        })
        .catch(error => {
            return serverError({
                body: {
                    success: false,
                    error: 'Erro interno',
                    details: error.message
                }
            });
        });
    } catch (error) {
        return serverError({
            body: {
                success: false,
                error: 'Erro interno',
                details: error.message
            }
        });
    }
}

// Função para segunda via
export function get_segundaVia(request) {
    try {
        const {billReceivableId, installmentId} = request.query;
        
        if (!billReceivableId || !installmentId) {
            return badRequest({
                body: {
                    success: false,
                    error: 'Parâmetros obrigatórios ausentes'
                }
            });
        }

        // Chamar API Sienge
        return fetch(`${API_CONFIG.baseURL}/payment-slip-notification?billReceivableId=${billReceivableId}&installmentId=${installmentId}`, {
            method: 'GET',
            headers: {
                'Authorization': `Basic ${Buffer.from(`${API_CONFIG.auth.username}:${API_CONFIG.auth.password}`).toString('base64')}`,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Erro na API: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Extrair URL do PDF
            let downloadUrl = null;
            if (data.results && data.results.length > 0) {
                downloadUrl = data.results[0].urlReport;
            }
            
            return ok({
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                body: {
                    success: true,
                    downloadUrl: downloadUrl,
                    digitableNumber: data.results && data.results.length > 0 ? data.results[0].digitableNumber : null
                }
            });
        })
        .catch(error => {
            return serverError({
                body: {
                    success: false,
                    error: 'Erro interno',
                    details: error.message
                }
            });
        });
    } catch (error) {
        return serverError({
            body: {
                success: false,
                error: 'Erro interno',
                details: error.message
            }
        });
    }
}