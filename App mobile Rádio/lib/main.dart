import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'screens/home_screen.dart';
import 'services/radio_service.dart';
import 'services/news_service.dart';
import 'services/podcast_service.dart';
import 'services/podcast_player_service.dart';
import 'services/theme_service.dart';
import 'utils/theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Força orientação portrait
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);
  
  // Sistema UI padrão que respeita navigation bar
  SystemChrome.setSystemUIOverlayStyle(
    SystemUiOverlayStyle.dark.copyWith(
      statusBarColor: Colors.transparent,
      systemNavigationBarColor: null, // Usa cor padrão do sistema
    ),
  );
  
  // Suprime erros do DebugService em desenvolvimento
  FlutterError.onError = (FlutterErrorDetails details) {
    if (details.toString().contains('DebugService')) {
      return;
    }
    FlutterError.presentError(details);
  };
  
  runApp(const RadioEntreRiosApp());
}

class RadioEntreRiosApp extends StatelessWidget {
  const RadioEntreRiosApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => RadioService()),
        ChangeNotifierProvider(create: (_) => NewsService()),
        ChangeNotifierProvider(create: (_) => PodcastService()),
        ChangeNotifierProvider(create: (_) => PodcastPlayerService()),
        ChangeNotifierProvider(create: (_) => ThemeService()),
      ],
      child: Consumer<ThemeService>(
        builder: (context, themeService, child) {
          return MaterialApp(
            title: 'Rádio Entre Rios',
            debugShowCheckedModeBanner: false,
            theme: themeService.isDarkMode ? AppTheme.darkTheme : AppTheme.lightTheme,
            home: const HomeScreen(),
          );
        },
      ),
    );
  }
}