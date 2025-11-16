# KAIMING FIX SIGNAL - 1756523501
# Este arquivo será lido pelo sacversion.py para aplicar fix

APPLY_KAIMING_FIX = True
TARGET_LAYERS = ["actor.latent_pi.0", "critic.qf0.0", "critic.qf1.0"]
INITIALIZATION = "kaiming_uniform"
NONLINEARITY = "leaky_relu"
REASON = "64.4% zeros no actor, 58.5% e 60.6% nos critics"
TIMESTAMP = 1756523501
