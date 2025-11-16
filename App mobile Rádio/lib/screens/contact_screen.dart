import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';
import '../services/theme_service.dart';
import '../utils/theme.dart';

class ContactScreen extends StatelessWidget {
  const ContactScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<ThemeService>(
      builder: (context, themeService, child) {
        return Scaffold(
          backgroundColor: themeService.isDarkMode
            ? AppTheme.darkBackground
            : AppTheme.lightBackground,
          appBar: AppBar(
            title: const Text('Sobre & Contato'),
            backgroundColor: themeService.isDarkMode
              ? AppTheme.darkBackground
              : AppTheme.lightBackground,
            foregroundColor: themeService.isDarkMode
              ? AppTheme.darkTextPrimary
              : AppTheme.lightTextPrimary,
            iconTheme: IconThemeData(
              color: themeService.isDarkMode
                ? AppTheme.darkTextPrimary
                : AppTheme.lightTextPrimary,
            ),
            elevation: 0,
          ),
          body: SingleChildScrollView(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildSectionTitle('Entre em Contato', themeService),
                const SizedBox(height: 16),
                _buildContactItem(
                  themeService,
              icon: Icons.phone,
              title: 'Telefone',
              subtitle: '(49) 3647-0292',
              onTap: () => _launchPhone('tel:+554936470292'),
            ),
                _buildContactItem(
                  themeService,
                  icon: Icons.email,
                  title: 'Email',
                  subtitle: 'portaria@radioentrerios.com.br',
                  onTap: () => _launchEmail('mailto:portaria@radioentrerios.com.br'),
                ),
                _buildContactItem(
                  themeService,
                  icon: Icons.chat,
                  title: 'WhatsApp Comercial',
                  subtitle: '(49) 99168-6860',
                  onTap: () => _launchWhatsApp('https://wa.me/5549991686860'),
                ),
                _buildContactItem(
                  themeService,
                  icon: Icons.location_on,
                  title: 'Endereço',
                  subtitle: 'Rua Visconde do Rio Branco, 1028\nCentro - Palmitos, SC',
                  onTap: null,
                ),

                const SizedBox(height: 32),
                _buildSectionTitle('Redes Sociais', themeService),
                const SizedBox(height: 16),
                _buildContactItem(
                  themeService,
                  icon: Icons.camera_alt,
                  title: 'Instagram',
                  subtitle: '@radioentrerios1055',
                  onTap: () => _launchUrl('https://www.instagram.com/radioentrerios1055'),
                ),

                const SizedBox(height: 32),
                _buildSectionTitle('Sobre a Rádio', themeService),
                const SizedBox(height: 16),
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: themeService.isDarkMode
                      ? AppTheme.darkCard
                      : AppTheme.lightCard,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: AppTheme.primaryOrange.withOpacity(0.3),
                    ),
                  ),
                  child: Text(
                    'A Rádio Entre Rios 105.5 FM é uma emissora de rádio localizada em Palmitos, Santa Catarina. '
                    'Oferecemos programação diversificada com notícias, música e entretenimento para toda a região. '
                    'Nossa missão é informar, entreter e conectar nossa comunidade através de conteúdo de qualidade.',
                    style: TextStyle(
                      fontSize: 16,
                      height: 1.5,
                      color: themeService.isDarkMode
                        ? AppTheme.darkTextPrimary
                        : AppTheme.lightTextPrimary,
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildSectionTitle(String title, ThemeService themeService) {
    return Text(
      title,
      style: const TextStyle(
        fontSize: 20,
        fontWeight: FontWeight.bold,
        color: AppTheme.primaryOrange,
      ),
    );
  }

  Widget _buildContactItem(
    ThemeService themeService, {
    required IconData icon,
    required String title,
    required String subtitle,
    VoidCallback? onTap,
  }) {
    return Card(
      color: themeService.isDarkMode
        ? AppTheme.darkCard
        : AppTheme.lightCard,
      elevation: themeService.isDarkMode ? 4 : 2,
      margin: const EdgeInsets.only(bottom: 12),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      child: ListTile(
        leading: Container(
          padding: const EdgeInsets.all(8),
          decoration: const BoxDecoration(
            color: AppTheme.primaryOrange,
            shape: BoxShape.circle,
          ),
          child: Icon(
            icon,
            color: Colors.white,
            size: 20,
          ),
        ),
        title: Text(
          title,
          style: TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 16,
            color: themeService.isDarkMode
              ? AppTheme.darkTextPrimary
              : AppTheme.lightTextPrimary,
          ),
        ),
        subtitle: Text(
          subtitle,
          style: TextStyle(
            color: themeService.isDarkMode
              ? AppTheme.darkTextSecondary
              : AppTheme.lightTextSecondary,
            fontSize: 14,
          ),
        ),
        onTap: onTap,
        trailing: onTap != null
            ? const Icon(
                Icons.arrow_forward_ios,
                size: 16,
                color: AppTheme.primaryOrange,
              )
            : null,
      ),
    );
  }

  Future<void> _launchPhone(String url) async {
    if (await canLaunchUrl(Uri.parse(url))) {
      await launchUrl(Uri.parse(url));
    }
  }

  Future<void> _launchEmail(String url) async {
    if (await canLaunchUrl(Uri.parse(url))) {
      await launchUrl(Uri.parse(url));
    }
  }

  Future<void> _launchWhatsApp(String url) async {
    if (await canLaunchUrl(Uri.parse(url))) {
      await launchUrl(Uri.parse(url), mode: LaunchMode.externalApplication);
    }
  }

  Future<void> _launchUrl(String url) async {
    if (await canLaunchUrl(Uri.parse(url))) {
      await launchUrl(Uri.parse(url), mode: LaunchMode.externalApplication);
    }
  }
}