// http-functions.js - HTTP Functions principais do Wix
import {ok, serverError, badRequest} from 'wix-http-functions';
import {fetch} from 'wix-fetch';
import wixData from 'wix-data';

// Configurações (inline para evitar problemas de import)
const CONFIG = {
    sienge: {
        base_url: 'https://api.sienge.com.br/arieproperties/public/api/v1',
        auth: {
            username: 'arieproperties-ti',
            password: 'eeoTCIXR9NtEdv118V1L16xZWZR64W7p'
        }
    },
    wix: {
        collection_id: 'Cliente'
    },
    password: {
        length: 8,
        characters: 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
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
            message: 'Backend Wix funcionando!',
            timestamp: new Date().toISOString(),
            domain: 'https://www.arieproperties.com'
        }
    });
}

// Função para buscar boletos
export async function get_boletosDetalhes(request) {
    try {
        const {cpfCnpj} = request.query;
        
        console.log(`🔍 Buscando boletos para CPF/CNPJ: ${cpfCnpj}`);
        
        if (!cpfCnpj) {
            return badRequest({
                headers: {
                    'Access-Control-Allow-Origin': '*'
                },
                body: {
                    success: false,
                    error: 'CPF/CNPJ é obrigatório'
                }
            });
        }

        // Criar autenticação Basic
        const credentials = Buffer.from(`${CONFIG.sienge.auth.username}:${CONFIG.sienge.auth.password}`).toString('base64');

        // Buscar saldo devedor atual
        const cpfLimpo = cpfCnpj.replace(/\D/g, '');
        const apiUrl = `${CONFIG.sienge.base_url}/current-debit-balance?cpf=${cpfLimpo}`;
        
        console.log(`🌐 Chamando API Sienge: ${apiUrl}`);
        
        const response = await fetch(apiUrl, {
            method: 'GET',
            headers: {
                'Authorization': `Basic ${credentials}`,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });

        console.log(`📡 Resposta da API Sienge - Status: ${response.status}`);

        if (!response.ok) {
            console.error(`❌ Erro HTTP da API Sienge: ${response.status}`);
            throw new Error(`Erro na API Sienge: ${response.status}`);
        }

        const data = await response.json();
        console.log(`📦 Dados recebidos da API:`, JSON.stringify(data, null, 2));
        
        // Verificar se a resposta tem a estrutura esperada
        if (!data || typeof data !== 'object') {
            console.error('❌ Resposta da API não é um objeto válido');
            throw new Error('Resposta inválida da API Sienge');
        }
        
        // Processar dados com verificações de segurança
        const boletosComDetalhes = [];
        
        // Verificar se results existe e é um array
        if (!data.results || !Array.isArray(data.results)) {
            console.log('⚠️ data.results não existe ou não é array:', data.results);
            return ok({
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                body: {
                    success: true,
                    boletos: [],
                    total: 0,
                    message: 'Nenhum boleto encontrado para este CPF/CNPJ'
                }
            });
        }
        
        console.log(`📋 Processando ${data.results.length} registros de bills`);
        
        for (let i = 0; i < data.results.length; i++) {
            const bill = data.results[i];
            
            try {
                console.log(`🔄 Processando bill ${i + 1}:`, bill);
                
                // Verificar se bill tem a estrutura básica
                if (!bill || typeof bill !== 'object') {
                    console.log(`⚠️ Bill ${i + 1} não é um objeto válido, pulando`);
                    continue;
                }
                
                const billReceivableId = bill.billReceivableId || null;
                const tiposParcelas = ['dueInstallments', 'payableInstallments', 'paidInstallments'];
                
                for (const tipo of tiposParcelas) {
                    // Verificar se o tipo de parcela existe e é um array
                    if (!bill[tipo] || !Array.isArray(bill[tipo])) {
                        console.log(`⚠️ ${tipo} não existe ou não é array em bill ${i + 1}`);
                        continue;
                    }
                    
                    console.log(`🔍 Processando ${bill[tipo].length} ${tipo}`);
                    
                    for (let j = 0; j < bill[tipo].length; j++) {
                        const installment = bill[tipo][j];
                        
                        try {
                            // Verificar se installment é válido
                            if (!installment || typeof installment !== 'object') {
                                console.log(`⚠️ Installment ${j + 1} em ${tipo} não é válido`);
                                continue;
                            }
                            
                            // Verificar se tem boleto gerado
                            const temBoleto = installment.generatedBoleto === true || installment.generatedBoleto === 'true';
                            
                            console.log(`💳 Installment ${j + 1} - generatedBoleto: ${installment.generatedBoleto}, temBoleto: ${temBoleto}`);
                            
                            if (temBoleto) {
                                const boletoData = {
                                    numero: installment.installmentNumber || `${i + 1}-${j + 1}`,
                                    valorOriginal: parseFloat(installment.originalValue) || 0,
                                    valorAtual: parseFloat(installment.currentBalance) || 0,
                                    valorPago: parseFloat(installment.paidValue) || 0,
                                    dataVencimento: installment.dueDate || null,
                                    situacao: determinarSituacao(installment, tipo),
                                    billReceivableId: billReceivableId,
                                    installmentId: installment.installmentId || null,
                                    podeSegundaVia: temBoleto && (parseFloat(installment.currentBalance) || 0) > 0,
                                    nossoNumero: installment.ourNumber || 'N/A',
                                    tipo: tipo
                                };
                                
                                boletosComDetalhes.push(boletoData);
                                console.log(`✅ Boleto adicionado:`, boletoData);
                            }
                        } catch (installmentError) {
                            console.error(`❌ Erro ao processar installment ${j + 1} em ${tipo}:`, installmentError);
                            // Continuar processando outros installments
                        }
                    }
                }
            } catch (billError) {
                console.error(`❌ Erro ao processar bill ${i + 1}:`, billError);
                // Continuar processando outras bills
            }
        }
        
        console.log(`✅ Total de boletos processados: ${boletosComDetalhes.length}`);
        
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

    } catch (error) {
        console.error('❌ Erro geral em boletosDetalhes:', error);
        console.error('❌ Stack trace:', error.stack);
        
        return serverError({
            headers: {
                'Access-Control-Allow-Origin': '*'
            },
            body: {
                success: false,
                error: 'Erro interno do servidor',
                details: error.message,
                cpfCnpj: request.query?.cpfCnpj || 'não informado'
            }
        });
    }
}

// Função auxiliar para determinar situação do boleto
function determinarSituacao(installment, tipo) {
    try {
        const valorAtual = parseFloat(installment.currentBalance) || 0;
        const valorPago = parseFloat(installment.paidValue) || 0;
        
        if (valorPago > 0 || valorAtual <= 0) {
            return 'Pago';
        }
        
        if (tipo === 'dueInstallments') {
            return 'Vencido';
        } else if (tipo === 'payableInstallments') {
            return 'A Vencer';
        } else if (tipo === 'paidInstallments') {
            return 'Pago';
        }
        
        return 'A Vencer';
    } catch (error) {
        console.error('❌ Erro ao determinar situação:', error);
        return 'Indefinido';
    }
}

// Função para segunda via
export async function get_segundaVia(request) {
    try {
        const {billReceivableId, installmentId} = request.query;
        
        if (!billReceivableId || !installmentId) {
            return badRequest({
                headers: {
                    'Access-Control-Allow-Origin': '*'
                },
                body: {
                    success: false,
                    error: 'Parâmetros billReceivableId e installmentId são obrigatórios'
                }
            });
        }

        // Criar autenticação Basic
        const credentials = Buffer.from(`${CONFIG.sienge.auth.username}:${CONFIG.sienge.auth.password}`).toString('base64');

        // Chamar API Sienge
        const apiUrl = `${CONFIG.sienge.base_url}/payment-slip-notification?billReceivableId=${billReceivableId}&installmentId=${installmentId}`;
        
        const response = await fetch(apiUrl, {
            method: 'GET',
            headers: {
                'Authorization': `Basic ${credentials}`,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`Erro na API Sienge: ${response.status}`);
        }

        const data = await response.json();
        
        // Extrair URL do PDF
        let downloadUrl = null;
        let digitableNumber = null;
        
        if (data.results && data.results.length > 0) {
            downloadUrl = data.results[0].urlReport;
            digitableNumber = data.results[0].digitableNumber;
        }
        
        return ok({
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: {
                success: true,
                downloadUrl: downloadUrl,
                digitableNumber: digitableNumber
            }
        });

    } catch (error) {
        console.error('Erro em segundaVia:', error);
        return serverError({
            headers: {
                'Access-Control-Allow-Origin': '*'
            },
            body: {
                success: false,
                error: 'Erro interno do servidor',
                details: error.message
            }
        });
    }
}

// Funções OPTIONS para CORS
export function options_boletosDetalhes(request) {
    return ok({
        headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        }
    });
}

export function options_segundaVia(request) {
    return ok({
        headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, OPTIONS', 
            'Access-Control-Allow-Headers': 'Content-Type'
        }
    });
}

export function options_teste(request) {
    return ok({
        headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        }
    });
}

// Função para gerar senha aleatória
function gerarSenha() {
    const caracteres = CONFIG.password.characters;
    let senha = '';
    for (let i = 0; i < CONFIG.password.length; i++) {
        senha += caracteres[Math.floor(Math.random() * caracteres.length)];
    }
    return senha;
}

// Função para importar clientes da Sienge
export async function get_importarClientes(request) {
    try {
        console.log('🔄 Iniciando importação de clientes da Sienge...');
        
        const credentials = Buffer.from(`${CONFIG.sienge.auth.username}:${CONFIG.sienge.auth.password}`).toString('base64');
        
        // Clientes separados por tipo de documento
        const clientesConhecidos = [
            // Pessoas Físicas (CPF)
            { documento: '37455407866', nome: 'THAIS CRISTINA JULIO BASTOS', email: 'thais.bastos01@hotmail.com' },
            { documento: '88770435804', nome: 'CARLOS AUGUSTO DE MATTOS LENCIONI', email: '' },
            
            // Pessoas Jurídicas (CNPJ) - dados diretos do arquivo JSON
            { documento: '37454492000103', nome: 'DOCSEG DOCUMENTACAO LEGAL LTDA.', email: 'CONTATO@DOCSEGLEGAL.COM.BR' },
            { documento: '54671694000118', nome: 'AR CHICAGO 002 SPE LTDA', email: '' },
            { documento: '53938696000168', nome: 'AR NEW YORK 001 SPE LTDA', email: '' },
            { documento: '50448249000132', nome: 'ARIE PROPERTIES S.A.', email: '' },
            { documento: '39526263000174', nome: 'Patagônia Capital Gestora de Recursos LTDA', email: 'rafael@arie.com.br' }
        ];
        
        let processados = 0;
        let importados = 0;
        let atualizados = 0;
        let erros = [];
        
        for (const cliente of clientesConhecidos) {
            try {
                const { documento, nome, email } = cliente;
                console.log(`🔍 Processando ${nome} (${documento})`);
                
                // Formatar documento (CPF ou CNPJ)
                let documentoFormatado;
                if (documento.length === 11) {
                    // CPF
                    documentoFormatado = documento.replace(/(\d{3})(\d{3})(\d{3})(\d{2})/, '$1.$2.$3-$4');
                } else {
                    // CNPJ
                    documentoFormatado = documento.replace(/(\d{2})(\d{3})(\d{3})(\d{4})(\d{2})/, '$1.$2.$3/$4-$5');
                }
                
                console.log(`👤 Processando cliente: ${nome}`);
                
                // Verificar se já existe no Wix
                const existente = await wixData.query(CONFIG.wix.collection_id)
                    .eq('cpfOuCnpj', documentoFormatado)
                    .find();
                
                const clienteData = {
                    title: nome, // Campo Title do Wix
                    nome: nome,
                    email: email || '',
                    cpfOuCnpj: documentoFormatado,
                    senha: gerarSenha()
                };
                
                if (existente.items.length > 0) {
                    // Atualizar cliente existente
                    const clienteId = existente.items[0]._id;
                    await wixData.update(CONFIG.wix.collection_id, {
                        _id: clienteId,
                        ...clienteData
                    });
                    atualizados++;
                    console.log(`✅ Cliente atualizado: ${nome}`);
                } else {
                    // Criar novo cliente
                    await wixData.save(CONFIG.wix.collection_id, clienteData);
                    importados++;
                    console.log(`✨ Cliente criado: ${nome}`);
                }
                
                processados++;
                
            } catch (error) {
                erros.push(`${cliente.nome}: ${error.message}`);
                console.error(`❌ Erro no cliente ${cliente.nome}:`, error);
            }
        }
        
        const resultado = {
            success: true,
            message: 'Importação concluída!',
            stats: {
                documentosProcessados: processados,
                clientesImportados: importados,
                clientesAtualizados: atualizados,
                totalErros: erros.length
            },
            erros: erros.length > 0 ? erros : undefined
        };
        
        console.log('📊 Resultado final:', resultado);
        
        return ok({
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: resultado
        });
        
    } catch (error) {
        console.error('💥 Erro geral na importação:', error);
        
        return serverError({
            headers: {
                'Access-Control-Allow-Origin': '*'
            },
            body: {
                success: false,
                error: 'Erro interno na importação',
                details: error.message
            }
        });
    }
}

// Função POST para importação com lista personalizada de CPFs
export async function post_importarClientes(request) {
    try {
        const {cpfs} = request.body || {};
        
        if (!cpfs || !Array.isArray(cpfs)) {
            return badRequest({
                headers: {
                    'Access-Control-Allow-Origin': '*'
                },
                body: {
                    success: false,
                    error: 'É necessário fornecer um array de CPFs no body: {"cpfs": ["123.456.789-00", ...]}'
                }
            });
        }
        
        console.log(`🔄 Importação personalizada iniciada com ${cpfs.length} CPFs`);
        
        const credentials = Buffer.from(`${CONFIG.sienge.auth.username}:${CONFIG.sienge.auth.password}`).toString('base64');
        
        let processados = 0;
        let importados = 0;
        let atualizados = 0;
        let erros = [];
        
        for (const cpf of cpfs) {
            try {
                const cpfLimpo = cpf.replace(/\D/g, '');
                const cpfFormatado = cpfLimpo.replace(/(\d{3})(\d{3})(\d{3})(\d{2})/, '$1.$2.$3-$4');
                
                console.log(`🔍 Processando CPF: ${cpfFormatado}`);
                
                // Buscar dados na Sienge
                const apiUrl = `${CONFIG.sienge.base_url}/current-debit-balance?cpf=${cpfLimpo}`;
                
                const response = await fetch(apiUrl, {
                    method: 'GET',
                    headers: {
                        'Authorization': `Basic ${credentials}`,
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    }
                });

                if (!response.ok) {
                    erros.push(`${cpfFormatado}: Erro HTTP ${response.status}`);
                    continue;
                }

                const data = await response.json();
                
                if (!data.results || data.results.length === 0) {
                    erros.push(`${cpfFormatado}: Cliente não encontrado na Sienge`);
                    continue;
                }
                
                const clienteInfo = data.results[0];
                const nome = clienteInfo.clientName || `Cliente ${cpfFormatado}`;
                
                // Verificar se já existe
                const existente = await wixData.query(CONFIG.wix.collection_id)
                    .eq('cpfOuCnpj', cpfFormatado)
                    .find();
                
                const clienteData = {
                    cpfOuCnpj: cpfFormatado,
                    nome: nome,
                    email: '',
                    telefone: '',
                    senha: gerarSenha(),
                    dataCreacao: new Date(),
                    ativo: true,
                    importadoEm: new Date()
                };
                
                if (existente.items.length > 0) {
                    const clienteId = existente.items[0]._id;
                    await wixData.update(CONFIG.wix.collection_id, {
                        _id: clienteId,
                        ...clienteData
                    });
                    atualizados++;
                } else {
                    await wixData.save(CONFIG.wix.collection_id, clienteData);
                    importados++;
                }
                
                processados++;
                
            } catch (error) {
                erros.push(`${cpf}: ${error.message}`);
            }
        }
        
        return ok({
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: {
                success: true,
                message: 'Importação personalizada concluída!',
                stats: {
                    cpfsEnviados: cpfs.length,
                    cpfsProcessados: processados,
                    clientesImportados: importados,
                    clientesAtualizados: atualizados,
                    totalErros: erros.length
                },
                erros: erros.length > 0 ? erros : undefined
            }
        });
        
    } catch (error) {
        return serverError({
            headers: {
                'Access-Control-Allow-Origin': '*'
            },
            body: {
                success: false,
                error: error.message
            }
        });
    }
}

// Função OPTIONS para importação
export function options_importarClientes(request) {
    return ok({
        headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        }
    });
}

// ============ SISTEMA DE LOGIN ============

// Função para fazer login
export async function post_login(request) {
    try {
        console.log('🔍 BACKEND - Fazendo parse do body...');
        const bodyData = await request.body.json();
        console.log('🔍 BACKEND - bodyData:', bodyData);
        
        const {email, senha, cpfCnpj} = bodyData || {};
        console.log('🔍 BACKEND - Dados extraídos - email:', email, 'senha:', senha, 'cpfCnpj:', cpfCnpj);
        
        console.log('🔍 VALIDAÇÃO - !email:', !email, '!cpfCnpj:', !cpfCnpj, '!senha:', !senha);
        console.log('🔍 VALIDAÇÃO - (!email && !cpfCnpj):', (!email && !cpfCnpj));
        console.log('🔍 VALIDAÇÃO - resultado final:', ((!email && !cpfCnpj) || !senha));
        
        if ((!email && !cpfCnpj) || !senha) {
            console.log('❌ VALIDAÇÃO FALHOU - retornando erro');
            return badRequest({
                headers: {
                    'Access-Control-Allow-Origin': '*'
                },
                body: {
                    success: false,
                    error: 'Email/CPF e senha são obrigatórios'
                }
            });
        }
        
        console.log('✅ VALIDAÇÃO PASSOU - continuando com login');
        
        // Buscar cliente na coleção
        let query = wixData.query(CONFIG.wix.collection_id);
        
        if (email) {
            console.log('🔎 Buscando por EMAIL:', email);
            query = query.eq('email', email);
        } else {
            console.log('🔎 Buscando por CPF no campo cpfOuCnpj:', cpfCnpj);
            query = query.eq('cpfOuCnpj', cpfCnpj);
        }
        
        const resultado = await query.find();
        
        if (resultado.items.length === 0) {
            return ok({
                headers: {
                    'Access-Control-Allow-Origin': '*'
                },
                body: {
                    success: false,
                    error: 'Cliente não encontrado'
                }
            });
        }
        
        const cliente = resultado.items[0];
        
        // Verificar senha
        if (cliente.senha !== senha) {
            console.log(`❌ Senha incorreta para: ${email || cpfCnpj}`);
            return ok({
                headers: {
                    'Access-Control-Allow-Origin': '*'
                },
                body: {
                    success: false,
                    error: 'Senha incorreta'
                }
            });
        }
        
        console.log(`✅ Login bem-sucedido: ${cliente.nome}`);
        
        // Login bem-sucedido
        return ok({
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: {
                success: true,
                message: 'Login realizado com sucesso',
                cliente: {
                    id: cliente._id,
                    nome: cliente.nome,
                    email: cliente.email,
                    cpfOuCnpj: cliente.cpfOuCnpj
                }
            }
        });
        
    } catch (error) {
        console.error('💥 Erro no login:', error);
        
        return serverError({
            headers: {
                'Access-Control-Allow-Origin': '*'
            },
            body: {
                success: false,
                error: 'Erro interno no servidor'
            }
        });
    }
}

// Função para alterar senha
export async function post_alterarSenha(request) {
    try {
        const {cpfCnpj, senhaAtual, novaSenha} = request.body || {};
        
        if (!cpfCnpj || !senhaAtual || !novaSenha) {
            return badRequest({
                headers: {
                    'Access-Control-Allow-Origin': '*'
                },
                body: {
                    success: false,
                    error: 'CPF/CNPJ, senha atual e nova senha são obrigatórios'
                }
            });
        }
        
        // Buscar cliente
        const resultado = await wixData.query(CONFIG.wix.collection_id)
            .eq('cpfOuCnpj', cpfCnpj)
            .find();
        
        if (resultado.items.length === 0) {
            return ok({
                headers: {
                    'Access-Control-Allow-Origin': '*'
                },
                body: {
                    success: false,
                    error: 'Cliente não encontrado'
                }
            });
        }
        
        const cliente = resultado.items[0];
        
        // Verificar senha atual
        if (cliente.senha !== senhaAtual) {
            return ok({
                headers: {
                    'Access-Control-Allow-Origin': '*'
                },
                body: {
                    success: false,
                    error: 'Senha atual incorreta'
                }
            });
        }
        
        // Atualizar senha
        await wixData.update(CONFIG.wix.collection_id, {
            _id: cliente._id,
            senha: novaSenha
        });
        
        console.log(`🔑 Senha alterada para: ${cliente.nome}`);
        
        return ok({
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: {
                success: true,
                message: 'Senha alterada com sucesso'
            }
        });
        
    } catch (error) {
        console.error('💥 Erro ao alterar senha:', error);
        
        return serverError({
            headers: {
                'Access-Control-Allow-Origin': '*'
            },
            body: {
                success: false,
                error: 'Erro interno no servidor'
            }
        });
    }
}

// Função para recuperar senha (gerar nova senha)
export async function post_recuperarSenha(request) {
    try {
        const {cpfCnpj, email} = request.body || {};
        
        if (!cpfCnpj && !email) {
            return badRequest({
                headers: {
                    'Access-Control-Allow-Origin': '*'
                },
                body: {
                    success: false,
                    error: 'CPF/CNPJ ou email é obrigatório'
                }
            });
        }
        
        // Buscar cliente
        let query = wixData.query(CONFIG.wix.collection_id);
        
        if (email) {
            query = query.eq('email', email);
        } else {
            query = query.eq('cpfOuCnpj', cpfCnpj);
        }
        
        const resultado = await query.find();
        
        if (resultado.items.length === 0) {
            return ok({
                headers: {
                    'Access-Control-Allow-Origin': '*'
                },
                body: {
                    success: false,
                    error: 'Cliente não encontrado'
                }
            });
        }
        
        const cliente = resultado.items[0];
        
        // Gerar nova senha
        const novaSenha = gerarSenha();
        
        // Atualizar no banco
        await wixData.update(CONFIG.wix.collection_id, {
            _id: cliente._id,
            senha: novaSenha
        });
        
        console.log(`🔄 Senha recuperada para: ${cliente.nome}`);
        
        // TODO: Aqui você pode adicionar envio de email com a nova senha
        
        return ok({
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: {
                success: true,
                message: 'Nova senha gerada com sucesso',
                novaSenha: novaSenha, // Em produção, envie por email
                cliente: {
                    nome: cliente.nome,
                    email: cliente.email
                }
            }
        });
        
    } catch (error) {
        console.error('💥 Erro ao recuperar senha:', error);
        
        return serverError({
            headers: {
                'Access-Control-Allow-Origin': '*'
            },
            body: {
                success: false,
                error: 'Erro interno no servidor'
            }
        });
    }
}

// Funções OPTIONS para CORS
export function options_login(request) {
    return ok({
        headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        }
    });
}

export function options_alterarSenha(request) {
    return ok({
        headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        }
    });
}

export function options_recuperarSenha(request) {
    return ok({
        headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        }
    });
}