const express = require('express');
const router = express.Router();
const orderController = require('../controllers/orderController');

/**
 * @route   POST /order
 * @desc    Criar um novo pedido
 * @access  Public
 */
router.post('/', orderController.createOrder);

/**
 * @route   GET /order/list
 * @desc    Listar todos os pedidos
 * @access  Public
 * @note    Esta rota deve vir ANTES da rota /:numeroPedido para evitar conflitos
 */
router.get('/list', orderController.listOrders);

/**
 * @route   GET /order/:numeroPedido
 * @desc    Obter um pedido específico
 * @access  Public
 */
router.get('/:numeroPedido', orderController.getOrder);

/**
 * @route   PUT /order/:numeroPedido
 * @desc    Atualizar um pedido
 * @access  Public
 */
router.put('/:numeroPedido', orderController.updateOrder);

/**
 * @route   DELETE /order/:numeroPedido
 * @desc    Deletar um pedido
 * @access  Public
 */
router.delete('/:numeroPedido', orderController.deleteOrder);

module.exports = router;
