const Order = require('../models/Order');

/**
 * Transforma os dados do formato de entrada para o formato do banco de dados
 */
const transformInputToDb = (inputData) => {
  return {
    orderId: inputData.numeroPedido,
    value: inputData.valorTotal,
    creationDate: new Date(inputData.dataCriacao),
    items: inputData.items.map(item => ({
      productId: parseInt(item.idItem),
      quantity: item.quantidadeItem,
      price: item.valorItem
    }))
  };
};

/**
 * Transforma os dados do banco para o formato de saída
 */
const transformDbToOutput = (dbData) => {
  return {
    numeroPedido: dbData.orderId,
    valorTotal: dbData.value,
    dataCriacao: dbData.creationDate,
    items: dbData.items.map(item => ({
      idItem: item.productId.toString(),
      quantidadeItem: item.quantity,
      valorItem: item.price
    }))
  };
};

/**
 * Criar um novo pedido
 * POST /order
 */
exports.createOrder = async (req, res) => {
  try {
    // Validação básica dos dados de entrada
    const { numeroPedido, valorTotal, dataCriacao, items } = req.body;

    if (!numeroPedido || !valorTotal || !dataCriacao || !items) {
      return res.status(400).json({
        error: 'Dados incompletos',
        message: 'numeroPedido, valorTotal, dataCriacao e items são obrigatórios'
      });
    }

    if (!Array.isArray(items) || items.length === 0) {
      return res.status(400).json({
        error: 'Items inválidos',
        message: 'Items deve ser um array com pelo menos um item'
      });
    }

    // Verificar se o pedido já existe
    const existingOrder = await Order.findOne({ orderId: numeroPedido });
    if (existingOrder) {
      return res.status(409).json({
        error: 'Pedido já existe',
        message: `O pedido ${numeroPedido} já está cadastrado`
      });
    }

    // Transformar dados para o formato do banco
    const orderData = transformInputToDb(req.body);

    // Criar o pedido
    const order = new Order(orderData);
    await order.save();

    // Retornar dados transformados
    const outputData = transformDbToOutput(order);

    res.status(201).json({
      message: 'Pedido criado com sucesso',
      data: outputData
    });
  } catch (error) {
    console.error('Erro ao criar pedido:', error);

    if (error.name === 'ValidationError') {
      return res.status(400).json({
        error: 'Erro de validação',
        message: error.message,
        details: Object.values(error.errors).map(err => err.message)
      });
    }

    res.status(500).json({
      error: 'Erro interno do servidor',
      message: 'Não foi possível criar o pedido'
    });
  }
};

/**
 * Obter um pedido específico por número
 * GET /order/:numeroPedido
 */
exports.getOrder = async (req, res) => {
  try {
    const { numeroPedido } = req.params;

    const order = await Order.findOne({ orderId: numeroPedido });

    if (!order) {
      return res.status(404).json({
        error: 'Pedido não encontrado',
        message: `O pedido ${numeroPedido} não existe`
      });
    }

    const outputData = transformDbToOutput(order);

    res.status(200).json({
      message: 'Pedido encontrado',
      data: outputData
    });
  } catch (error) {
    console.error('Erro ao buscar pedido:', error);
    res.status(500).json({
      error: 'Erro interno do servidor',
      message: 'Não foi possível buscar o pedido'
    });
  }
};

/**
 * Listar todos os pedidos
 * GET /order/list
 */
exports.listOrders = async (req, res) => {
  try {
    // Parâmetros de paginação (opcional)
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;
    const skip = (page - 1) * limit;

    const orders = await Order.find()
      .sort({ creationDate: -1 })
      .skip(skip)
      .limit(limit);

    const total = await Order.countDocuments();

    const outputData = orders.map(order => transformDbToOutput(order));

    res.status(200).json({
      message: 'Pedidos listados com sucesso',
      data: outputData,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit)
      }
    });
  } catch (error) {
    console.error('Erro ao listar pedidos:', error);
    res.status(500).json({
      error: 'Erro interno do servidor',
      message: 'Não foi possível listar os pedidos'
    });
  }
};

/**
 * Atualizar um pedido
 * PUT /order/:numeroPedido
 */
exports.updateOrder = async (req, res) => {
  try {
    const { numeroPedido } = req.params;

    // Verificar se o pedido existe
    const existingOrder = await Order.findOne({ orderId: numeroPedido });
    if (!existingOrder) {
      return res.status(404).json({
        error: 'Pedido não encontrado',
        message: `O pedido ${numeroPedido} não existe`
      });
    }

    // Transformar dados para o formato do banco
    const orderData = transformInputToDb(req.body);

    // Atualizar o pedido
    const updatedOrder = await Order.findOneAndUpdate(
      { orderId: numeroPedido },
      orderData,
      { new: true, runValidators: true }
    );

    const outputData = transformDbToOutput(updatedOrder);

    res.status(200).json({
      message: 'Pedido atualizado com sucesso',
      data: outputData
    });
  } catch (error) {
    console.error('Erro ao atualizar pedido:', error);

    if (error.name === 'ValidationError') {
      return res.status(400).json({
        error: 'Erro de validação',
        message: error.message,
        details: Object.values(error.errors).map(err => err.message)
      });
    }

    res.status(500).json({
      error: 'Erro interno do servidor',
      message: 'Não foi possível atualizar o pedido'
    });
  }
};

/**
 * Deletar um pedido
 * DELETE /order/:numeroPedido
 */
exports.deleteOrder = async (req, res) => {
  try {
    const { numeroPedido } = req.params;

    const deletedOrder = await Order.findOneAndDelete({ orderId: numeroPedido });

    if (!deletedOrder) {
      return res.status(404).json({
        error: 'Pedido não encontrado',
        message: `O pedido ${numeroPedido} não existe`
      });
    }

    res.status(200).json({
      message: 'Pedido deletado com sucesso',
      data: {
        numeroPedido: deletedOrder.orderId
      }
    });
  } catch (error) {
    console.error('Erro ao deletar pedido:', error);
    res.status(500).json({
      error: 'Erro interno do servidor',
      message: 'Não foi possível deletar o pedido'
    });
  }
};
