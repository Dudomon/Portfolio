import 'package:flutter/material.dart';

class ResponsiveUtils {
  static double getScreenWidth(BuildContext context) {
    return MediaQuery.of(context).size.width;
  }

  static double getScreenHeight(BuildContext context) {
    return MediaQuery.of(context).size.height;
  }

  static bool isMobile(BuildContext context) {
    return getScreenWidth(context) < 600;
  }

  static bool isTablet(BuildContext context) {
    return getScreenWidth(context) >= 600 && getScreenWidth(context) < 1024;
  }

  static bool isDesktop(BuildContext context) {
    return getScreenWidth(context) >= 1024;
  }

  static double getHorizontalPadding(BuildContext context) {
    if (isMobile(context)) {
      return getScreenWidth(context) * 0.04;
    } else if (isTablet(context)) {
      return getScreenWidth(context) * 0.06;
    } else {
      return getScreenWidth(context) * 0.08;
    }
  }

  static double getVerticalPadding(BuildContext context) {
    return getScreenHeight(context) * 0.02;
  }

  static double getCardHeight(BuildContext context) {
    if (isMobile(context)) {
      return getScreenHeight(context) * 0.15;
    } else if (isTablet(context)) {
      return getScreenHeight(context) * 0.12;
    } else {
      return getScreenHeight(context) * 0.10;
    }
  }

  static double getFontSize(BuildContext context, double baseSize) {
    final screenWidth = getScreenWidth(context);
    if (screenWidth < 320) {
      return baseSize * 0.85;
    } else if (screenWidth < 375) {
      return baseSize * 0.9;
    } else if (screenWidth < 414) {
      return baseSize;
    } else if (screenWidth < 600) {
      return baseSize * 1.1;
    } else {
      return baseSize * 1.2;
    }
  }
}