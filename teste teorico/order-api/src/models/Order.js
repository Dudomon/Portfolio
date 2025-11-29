const mongoose = require('mongoose');

const itemSchema = new mongoose.Schema({
  productId: {
    type: Number,
    required: [true, 'Product ID is required']
  },
  quantity: {
    type: Number,
    required: [true, 'Quantity is required'],
    min: [1, 'Quantity must be at least 1']
  },
  price: {
    type: Number,
    required: [true, 'Price is required'],
    min: [0, 'Price must be positive']
  }
}, { _id: false });

const orderSchema = new mongoose.Schema({
  orderId: {
    type: String,
    required: [true, 'Order ID is required'],
    unique: true,
    trim: true
  },
  value: {
    type: Number,
    required: [true, 'Value is required'],
    min: [0, 'Value must be positive']
  },
  creationDate: {
    type: Date,
    required: [true, 'Creation date is required'],
    default: Date.now
  },
  items: {
    type: [itemSchema],
    required: [true, 'Items are required'],
    validate: {
      validator: function(items) {
        return items && items.length > 0;
      },
      message: 'Order must have at least one item'
    }
  }
}, {
  timestamps: true,
  versionKey: false
});

// Index para busca mais rápida
orderSchema.index({ orderId: 1 });

// Método para transformar o objeto antes de retornar
orderSchema.methods.toJSON = function() {
  const obj = this.toObject();
  delete obj._id;
  return obj;
};

const Order = mongoose.model('Order', orderSchema);

module.exports = Order;
