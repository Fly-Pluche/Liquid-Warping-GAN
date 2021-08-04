

为前端（HTML5+CSS+JS）的学习笔记

## W3school

实现HTML代码编译的在线网站：https://www.w3school.com.cn/tiy/t.asp?f=eg_html_input_name

![image-20210714144841247](C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210714144841247.png)

## </br>

换行

![image-20210714145057158](C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210714145057158.png)

<img src="C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210714145120108.png" alt="image-20210714145120108" style="zoom: 33%;" />

## onfocus()事件，onblur()事件，onload()事件（JS）

### onfocus（聚焦）

事件在对象获得焦点时发生，即鼠标点击input的输入窗口的时候，触发。

实例：使输入框变黄

```html
<html>
<head>
<script type="text/javascript">
    function setStyle(x)
    {
        document.getElementById(x).style.background="yellow"
    }
</script>
</head>
<body>
    First name: <input type="text" onfocus="setStyle(this.id)" id="fname" /><br />
    Last name: <input type="text" onfocus="setStyle(this.id)" id="lname" />
</body>
</html>
```

### onblur(模糊，失焦)



```html
<html>
<head>
<script type="text/javascript">
    function upperCase()
    {
        var x=document.getElementById("fname").value
        document.getElementById("fname").value=x.toUpperCase()    //toUpperCase() 方法用于把字符串转换为大写
    } 
</script>
</head>
<body> 
    输入您的姓名： <input type="text" id="fname" onblur="upperCase()"/> 
</body>
</html>
```



> 来自：[csdn—–木木木华](https://blog.csdn.net/catascdd/article/details/79993580?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522162623923616780271566211%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=162623923616780271566211&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-2-79993580.pc_search_result_cache&utm_term=onfocus)

## 定义函数

function name(){
}

## 连接额外的css文件

<head>
    <link rel="stylesheet" href="style.css">
</head>

其中rel是relationship，描述当前的页面与href所指定文档的关系。至于href就是相关联的css文件路径。

## <title>标签

![image-20210716145419289](C:\Users\BlackFriday\AppData\Roaming\Typora\typora-user-images\image-20210716145419289.png)

<title> document </title>



