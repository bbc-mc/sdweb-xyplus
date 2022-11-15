# X/Y Plus

X/Y plot の機能追加版

## X-Y の実行順序を選択: Radio

- "Start from Axis

## 実行後に Checkpoint を元に戻す: Checkbox

## X-Y Grid に表示される Checkpoint 名について、別名を表示する: Enabled

- "Checkpoint Name" を選択

- x_type または y_type に Checkpoint Name を入力

- `#` 以降に、別名を入力
  
  ```
  sd-v1.4 # SD14
  wd-v1.3 # WD13
  ```

## 保存される grid の PNG ファイルに、seed などの PNG Info 情報を追加できる: Enabled

## X/Y の選択肢について、(私が)よく使うものだけを表示する: Checkbox

## 数値入力について、より柔軟な range, step 入力に対応

- seed and step and range
  "   123 ( 4 ) [ 5 ] "  => "123, 127, 131, 135, 139"
  " - 123 ( 4 ) [ 5 ] "  => "123, 119, 115, 111, 107"
- seed and range
  "   123 [ 5 ] "        => "123, 124, 125, 126, 127"
  " - 123 [ 5 ] "        => "123, 122, 121, 120, 119"
- step and range
   - seed値は、デフォルトUIにある Seed 項の値を参照します
     "   ( 4 ) [ 5 ] "      => "<seed>, <>+4, <>+8, <>+12, <>+16"
     " - ( 4 ) [ 5 ] "      => "<seed>, <>-4, <>-8, <>-12, <>-16"
- range
   - seed値は、デフォルトUIにある Seed 項の値を参照します
     "   [ 5 ] "            => "<seed>, +1, +2, +3, +4"
     " - [ 5 ] "            => "<seed>, -1, -2, -3, -4"
