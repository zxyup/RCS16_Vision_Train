# SSH与Github
## SSH简介
### [图解SSH原理](https://www.jianshu.com/p/33461b619d53)

## Github入门
###  1.SSH关联到远程仓库
- 检查本地主机是否已经存在ssh key
- 生成ssh key
- 获取ssh key公钥内容（id_rsa.pub）
- Github账号上添加公钥
- 验证是否设置成功
###  2.个人代码托管

1. 初始化：本地建库（即文件夹），git init
2. 添加到仓库：代码文件放入本地库，git add .
3. 提交： git commit -m “注释内容”，提交到仓库
4. 新建远程仓库并关联：在Github上设置好SSH密钥后，新建一个远程仓库， git remote add origin https://github.com/xu-xiaoya/Elegent.git关联
5. 推送：git push (-u) origin master，把本地仓库的代码推送到远程仓库Github上
### 3.多人协作
- git clone
- 创建自己的分支
- 提交到远程仓库
- compare and pull request

##### 参考资料
- [Github配置ssh key的步骤](https://blog.csdn.net/weixin_42310154/article/details/118340458)
- [git本地仓库建立与远程连接](https://blog.csdn.net/qq_29493173/article/details/113094143)
- [git创建分支，提交代码详细流程](https://blog.csdn.net/weixin_43367262/article/details/100575221)
- [git push -u origin master 与git push --set-upstream origin master](https://blog.csdn.net/qq_29493173/article/details/113094143)