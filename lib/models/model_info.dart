class ModelInfo {
  final String id;
  final String title;
  final String fileName;
  final String url;
  final String? sha256;
  final int? sizeBytes;

  const ModelInfo({
    required this.id,
    required this.title,
    required this.fileName,
    required this.url,
    this.sha256,
    this.sizeBytes,
  });
}

class OcrModelCatalog {
  static const base = 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2';

  // Official RapidOCR PP-OCRv5 mobile ONNX model + Arabic/Persian dictionary.
  static const det = ModelInfo(
    id: 'ppocrv5_det_mobile',
    title: 'مدل تشخیص متن PP-OCRv5',
    fileName: 'ch_PP-OCRv5_det_mobile.onnx',
    url: '$base/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx',
    sha256: '4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae',
  );

  static const rec = ModelInfo(
    id: 'ppocrv5_arabic_rec_mobile',
    title: 'مدل تشخیص حروف فارسی/عربی PP-OCRv5',
    fileName: 'arabic_PP-OCRv5_rec_mobile.onnx',
    url: '$base/onnx/PP-OCRv5/rec/arabic_PP-OCRv5_rec_mobile.onnx',
    sha256: 'c1192e632d0baa9146ae5b756a0e635e3dc63c1733737ebfd1629e87144e9295',
  );

  static const cls = ModelInfo(
    id: 'ppocrv5_cls_mobile',
    title: 'مدل تشخیص جهت متن',
    fileName: 'ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx',
    url: '$base/onnx/PP-OCRv5/cls/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx',
    sha256: '54379ae5174d026780215fc748a7f31910dee36818e63d49e17dc598ecc82df7',
  );

  static const dict = ModelInfo(
    id: 'ppocrv5_arabic_dict',
    title: 'فرهنگ حروف فارسی/عربی',
    fileName: 'ppocrv5_arabic_dict.txt',
    url: '$base/paddle/PP-OCRv5/rec/arabic_PP-OCRv5_rec_mobile/ppocrv5_arabic_dict.txt',
  );

  static const all = [det, rec, cls, dict];
}
