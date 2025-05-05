## 📋 AI_For_Coloring

### 🚀 Установка зависимостей

Для работы проекта необходимо установить все зависимости. Выполните следующие шаги в вашем терминале или в PyCharm.

#### 1. Обновление `pip` и установка базовых утилит
Обновите `pip` и установите `setuptools`, чтобы избежать проблем совместимости:

```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
pip install setuptools
```

#### 2. Установка основных зависимостей
Установите библиотеки, необходимые для работы бота и генерации раскрасок:

```bash
pip install aiogram==3.13.1 diffusers==0.30.3 transformers==4.45.2 opencv-python==4.11.0.86 numpy==1.26.4 deep-translator==1.11.4 pydantic-settings==2.5.2
```

#### 3. Установка PyTorch
Для ускорения генерации изображений через Stable Diffusion рекомендуется использовать GPU. Установите PyTorch с поддержкой CUDA:

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

> **Примечание**: Убедитесь, что версия CUDA (например, `cu121`) соответствует установленной на вашем компьютере. Если вы не используете GPU, установите версию для CPU:
>
> ```bash
> pip install torch==2.4.1 torchvision==0.19.1
> ```

#### 4. Установка через `requirements.txt` (альтернативный способ)
Вы можете использовать файл `requirements.txt` для установки всех зависимостей одной командой:

```bash
pip install -r requirements.txt
```

Содержимое `requirements.txt`:
```
aiogram==3.13.1
diffusers==0.30.3
transformers==4.45.2
opencv-python==4.11.0.86
numpy==1.26.4
deep-translator==1.11.4
pydantic-settings==2.5.2
torch==2.4.1
torchvision==0.19.1
```

#### 5. Проверка установки
Убедитесь, что все зависимости установлены корректно:

```bash
pip list
```

#### ⚠️ Возможные проблемы
- Если вы используете Python 3.12+, могут возникнуть ошибки с `distutils`. Убедитесь, что `setuptools` установлен.
- Убедитесь, что у вас достаточно места для кэша Hugging Face (по умолчанию `D:/huggingface_cache`).

---
